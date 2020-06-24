import torch

import os
import time
from torch import autograd

from blox.basic_types import map_dict
from gcp.rec_planner_utils.checkpointer import CheckpointHandler
from blox.utils import dummy_context, RecursiveAverageMeter
from blox import AttrDict
from gcp.rec_planner_utils.gcp_trainer import GCPTrainer
from blox.utils import AverageMeter

import warnings
warnings.simplefilter('once')


class ModelTrainer(GCPTrainer):
    """ This class defines the training loop of the GCP model"""
    
    def train(self, start_epoch):
        if not self.args.skip_first_val:
            self.val()
            
        for epoch in range(start_epoch, self._hp.num_epochs):
            self.train_epoch(epoch)
        
            if not self.args.dont_save:
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
        return self.global_step % self.log_images_interval == 0 and self.args.log_images
    
    @property
    def log_outputs_now(self):
        return self.global_step % self.args.log_outputs_interval == 0 or self.global_step % self.log_images_interval == 0

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
        self.log_images_interval = int(epoch_len / self.args.imepoch)
        
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
            
            if self.args.train_loop_pdb:
                import pdb; pdb.set_trace()
            
            upto_log_time.update(time.time() - end)
            if self.log_outputs_now and not self.args.dont_save:
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
                if self.args.verbose_timing: print("avg FW/BW time: {:.3f}s/batch".format(forward_backward_time.avg))
            
            del output, losses
            self.global_step = self.global_step + 1

    def val(self, test_control=True):
        print('Running Testing')
        if self.args.test_prediction:
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
                
                if not self.args.dont_save:
                    if self.evaluator is not None:
                        self.evaluator.dump_results(self.global_step)
                    if self.run_testmetrics:
                        print("Finished Evaluation! Exiting...")
                        exit(0)
    
                    self.model.log_outputs(
                        output, inputs, losses_meter.avg, self.global_step, log_images=self.args.log_images, phase='val')
                    print(('\nTest set: Average loss: {:.4f} in {:.2f}s\n'
                           .format(losses_meter.avg.total.value.item(), time.time() - start)))
                    if self.args.verbose_timing: print("avg Inference time: {:.3f}s/batch".format(infer_time.avg))
            del output

        
if __name__ == '__main__':
    ModelTrainer()
