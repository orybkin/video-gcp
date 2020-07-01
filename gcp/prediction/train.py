import os
import time
import warnings
import math
import numpy as np
from copy import deepcopy
from tensorboardX import SummaryWriter

import torch
from torch import autograd

from blox import AttrDict
from blox.basic_types import map_dict
from blox.utils import AverageMeter
from blox.utils import RecursiveAverageMeter
from gcp.prediction.training.checkpoint_handler import CheckpointHandler, NoCheckpointsException
from gcp.prediction.training.gcp_builder import GCPBuilder
from gcp.evaluation.compute_metrics import Evaluator

warnings.simplefilter('once')


class ModelTrainer(GCPBuilder):
    """ This class defines the training loop of the GCP model"""

    def run(self):
        """ Runs training """
        args = self.cmd_args
        model_conf = self.model._hp
        data_conf = self.train_loader.dataset.data_conf
    
        # TODO clean up resuming
        self.global_step = 0
        start_epoch = 0
        if args.resume or ('checkpt_path' in model_conf and model_conf.checkpt_path is not None):
            ckpt_path = model_conf.checkpt_path if 'checkpt_path' in model_conf else None
            start_epoch = self.resume(args.resume, ckpt_path)
    
        if args.val_sweep:
            epochs = CheckpointHandler.get_epochs(os.path.join(self._hp.exp_path, 'weights'))
            for epoch in list(sorted(epochs))[::4]:
                self.resume(epoch)
                self.val()
            return
    
        if args.dataset_val_sweep:
            self.run_dataset_val_sweep(args, data_conf, model_conf)
            return
    
        ## Train
        if args.train:
            self.train(start_epoch)
        else:
            self.val()

    def resume(self, ckpt, path=None):
        path = os.path.join(self._hp.exp_path, 'weights') if path is None else os.path.join(path, 'weights')
        assert ckpt is not None  # need to specify resume epoch for loading checkpoint
        try:
            weights_file = CheckpointHandler.get_resume_ckpt_file(ckpt, path)
        except NoCheckpointsException:
            return 0
        self.global_step, start_epoch, _ = \
            CheckpointHandler.load_weights(weights_file, self.model,
                                           load_step_and_opt=True, optimizer=self.optimizer,
                                           dataset_length=len(self.train_loader) * self._hp.batch_size,
                                           strict=self.cmd_args.strict_weight_loading)
        self.model.to(self.model.device)
        return start_epoch

    def run_dataset_val_sweep(self, args, data_conf, model_conf):
        assert 'sweep_specs' in data_conf.dataset_spec and data_conf.dataset_spec.sweep_specs  # need to define sweep_specs
        for sweep_spec in data_conf.dataset_spec.sweep_specs:
            print("\nStart eval of dataset {}...".format(sweep_spec.name))
            dc, mc = deepcopy(data_conf), deepcopy(model_conf)
            dc.dataset_spec.dataset_class = sweep_spec.dataset_class
            dc.dataset_spec.split = sweep_spec.split
            dc.dataset_spec.max_seq_len = sweep_spec.max_seq_len
            mc.update({i: dc.dataset_spec[i] for i in dc.dataset_spec if not isinstance(dc.dataset_spec[i], list)})
            if "dense_rec_type" not in mc or mc["dense_rec_type"] is not "svg":
                mc["hierarchy_levels"] = int(np.ceil(math.log2(sweep_spec.max_seq_len)))
            log_dir = self.log_dir + "_" + sweep_spec.name
            writer = SummaryWriter(log_dir)
        
            def rebuild_phase(logger, model, phase, n_repeat=1, dataset_size=-1):
                logger = logger(log_dir, self._hp, max_seq_len=sweep_spec.max_seq_len, summary_writer=writer)
                model = model(mc, logger).to(self.device)
                model.device = self.device
                loader = self.get_dataset(args, model, dc, phase, n_repeat, dataset_size)
                return logger, model, loader
        
            self.logger, self.model, self.val_loader = \
                rebuild_phase(self._hp.logger, self._hp.model, "val", dataset_size=args.val_data_size)
            self.evaluator = Evaluator(self.model, log_dir, self._hp, self.cmd_args.metric, self.logger)
            if args.resume:
                self.resume(args.resume)
            else:
                self.resume("latest")
            self.val()
        print("...Done!")

    def train(self, start_epoch):
        if not self.cmd_args.skip_first_val:
            self.val()
            
        for epoch in range(start_epoch, self._hp.num_epochs):
            self.train_epoch(epoch)
        
            if not self.cmd_args.dont_save:
                self.save_checkpoint(epoch)
            self.val(not (epoch - start_epoch) % 3)
            
    def save_checkpoint(self, epoch):
        state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        folder = os.path.join(self._hp.exp_path, 'weights')
        os.makedirs(folder, exist_ok=True)
        torch.save(state, os.path.join(folder, CheckpointHandler.get_ckpt_name(epoch)))
    
    @property
    def log_images_now(self):
        return self.global_step % self.log_images_interval == 0 and self.cmd_args.log_images
    
    @property
    def log_outputs_now(self):
        return self.global_step % self.cmd_args.log_outputs_interval == 0 or self.global_step % self.log_images_interval == 0

    def try_move_to_dev(self, data):
        try:
            return data.to(self.device)
        except:
            # print('warning: could not move {} to gpu'.format(type(data)))
            return data

    def train_epoch(self, epoch):
        self.model.train()
        epoch_len = len(self.train_loader)
        end = time.time()
        batch_time = AverageMeter()
        upto_log_time = AverageMeter()
        data_load_time = AverageMeter()
        forward_backward_time = AverageMeter()
        self.log_images_interval = int(epoch_len / self.cmd_args.imepoch)
        
        print('starting epoch ', epoch)
        
        for self.batch_idx, sample_batched in enumerate(self.train_loader):
            data_load_time.update(time.time() - end)
            inputs = AttrDict(map_dict(self.try_move_to_dev, sample_batched))
            with self.training_context():
                self.optimizer.zero_grad()
                start_fw_bw = time.time()
                output = self.model(inputs)
                losses = self.model.loss(inputs, output)
                losses.total = self.model.get_total_loss(inputs, losses)
                losses.total.value.backward()
                self.call_hooks(inputs, output, losses, epoch)
                self.optimizer.step()
                self.model.step()
                forward_backward_time.update(time.time() - start_fw_bw)
            
            if self.cmd_args.train_loop_pdb:
                import pdb; pdb.set_trace()
            
            upto_log_time.update(time.time() - end)
            if self.log_outputs_now and not self.cmd_args.dont_save:
                self.model.log_outputs(output, inputs, losses, self.global_step,
                                       log_images=self.log_images_now, phase='train')
            batch_time.update(time.time() - end)
            end = time.time()
            
            if self.log_outputs_now:
                print('GPU {}: {}'.format(os.environ["CUDA_VISIBLE_DEVICES"] if self.use_cuda else 'none', self._hp.exp_path))
                print(('itr: {} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.global_step, epoch, self.batch_idx, len(self.train_loader),
                    100. * self.batch_idx / len(self.train_loader), losses.total.value.item())))
                
                print('avg time for loading: {:.2f}s, logs: {:.2f}s, compute: {:.2f}s, total: {:.2f}s'
                      .format(data_load_time.avg,
                              batch_time.avg - upto_log_time.avg,
                              upto_log_time.avg - data_load_time.avg,
                              batch_time.avg))
                togo_train_time = batch_time.avg * (self._hp.num_epochs - epoch) * epoch_len / 3600.
                print('ETA: {:.2f}h'.format(togo_train_time))
                if self.cmd_args.verbose_timing: print("avg FW/BW time: {:.3f}s/batch".format(forward_backward_time.avg))
            
            del output, losses
            self.global_step = self.global_step + 1

    def val(self, test_control=True):
        print('Running Testing')
        if self.cmd_args.test_prediction:
            start = time.time()
            losses_meter = RecursiveAverageMeter()
            infer_time = AverageMeter()

            # self.model.eval()
            with autograd.no_grad():
                for batch_idx, sample_batched in enumerate(self.val_loader):
                    inputs = AttrDict(map_dict(self.try_move_to_dev, sample_batched))
                    with self.model.val_mode(pred_length=False):
                        infer_start = time.time()
                        output = self.model(inputs, 'test')
                        infer_time.update(time.time() - infer_start)
                        if self.evaluator is not None:    # force eval on all batches for reduced noise
                            self.evaluator.eval(inputs, output, self.model)
                    # run train model to get NLL on validation data
                    output_train_mdl = self.model(inputs)
                    losses = self.model.loss(inputs, output_train_mdl)
                    losses.total = self.model.get_total_loss(inputs, losses)
                    losses_meter.update(losses)
                    del losses
                    del output_train_mdl

                    # if batch_idx == 0:
                    #     break
                
                if not self.cmd_args.dont_save:
                    if self.evaluator is not None:
                        self.evaluator.dump_results(self.global_step)
                    if self.cmd_args.metric:
                        print("Finished Evaluation! Exiting...")
                        exit(0)
    
                    self.model.log_outputs(
                        output, inputs, losses_meter.avg, self.global_step, log_images=self.cmd_args.log_images, phase='val')
                    print(('\nTest set: Average loss: {:.4f} in {:.2f}s\n'
                           .format(losses_meter.avg.total.value.item(), time.time() - start)))
                    if self.cmd_args.verbose_timing: print("avg Inference time: {:.3f}s/batch".format(infer_time.avg))
            del output

        
if __name__ == '__main__':
    trainer = ModelTrainer()
    trainer.run()
