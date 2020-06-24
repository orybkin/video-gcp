import math
from copy import deepcopy

import matplotlib; matplotlib.use('Agg')
import torch
import dload
import argparse
import os
import time
from shutil import copy
import datetime
import imp
from tensorflow.contrib.training import HParams
from tensorboardX import SummaryWriter
import numpy as np
from torch import autograd
from torch.optim import Adam, RMSprop, SGD
from functools import partial

from gcp.datasets.data_loader import FolderSplitVarLenVideoDataset
from gcp.rec_planner_utils.checkpointer import CheckpointHandler, save_cmd, save_git, get_config_path,\
    NoCheckpointsException
from blox.utils import dummy_context, RecursiveAverageMeter
from blox import AttrDict
from blox.torch.training import LossSpikeHook, NanGradHook, NoneGradHook, DataParallelWrapper, \
    get_clipped_optimizer
from gcp.run_control_experiment import ControlManager
from gcp.evaluation.compute_metrics import Evaluator
from gcp.rec_planner_utils.trainer_base import BaseTrainer
from gcp.rec_planner_utils import global_params
from gcp.rec_planner_utils import vis_utils
from blox.torch.radam import RAdam


def save_config(conf_path, exp_conf_path):
    copy(conf_path, exp_conf_path)

def get_exp_dir():
    return os.environ['GCP_EXP_DIR']


def datetime_str():
    return datetime.datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")


def make_path(exp_dir, conf_path, prefix, make_new_dir):
    # extract the subfolder structure from config path
    path = conf_path.split('experiments/', 1)[1]
    if make_new_dir:
        prefix += datetime_str()
    base_path = os.path.join(exp_dir, path)
    return os.path.join(base_path, prefix) if prefix else base_path


def set_seeds(seed=0):
    """Sets all seeds and disables non-determinism in cuDNN backend."""
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def get_dataset_path(dataset_name):
    """Returns path to dataset."""
    return os.path.join(os.environ["GCP_DATA_DIR"], dataset_name)


def download_data(dataset_name):
    """Downloads data if not yet existent."""
    DATA_URLs = AttrDict(
        nav_9rooms='https://www.seas.upenn.edu/~oleh/datasets/gcp/nav_9rooms.zip',
        nav_25rooms='https://www.seas.upenn.edu/~oleh/datasets/gcp/nav_25rooms.zip',
        sawyer='https://www.seas.upenn.edu/~oleh/datasets/gcp/sawyer.zip',
        h36m='https://www.seas.upenn.edu/~oleh/datasets/gcp/h36m.zip',
    )
    if dataset_name not in DATA_URLs:
        raise ValueError("Dataset identifier {} is not known!".format(dataset_name))
    if not os.path.exists(get_dataset_path(dataset_name)):
        print("Downloading dataset from {} to {}.".format(DATA_URLs[dataset_name], os.environ["GCP_DATA_DIR"]))
        print("This may take a few minutes...")
        dload.save_unzip(DATA_URLs[dataset_name], os.environ["GCP_DATA_DIR"], delete_after=True)
        print("...Done!")
    
    
class GCPTrainer(BaseTrainer):
    """ This class constructs the GCP model, dataset and the optimizers """

    def __init__(self):
        self.batch_idx = 0
    
        ## Set up params
        args, conf_module, conf, model_conf, exp_dir, conf_path = self.get_configs()
    
        self._hp = self._default_hparams()
        self.override_defaults(conf)  # override defaults with config file

        # download data, get data config
        download_data(conf.dataset_name)
        data_conf = self.get_data_config(conf_module)
    
        self._hp.set_hparam('exp_path', make_path(exp_dir, args.path, args.prefix, args.new_dir))
        self.log_dir = log_dir = os.path.join(self._hp.exp_path, 'events')
        print('using log dir: ', log_dir)
    
        self.run_testmetrics = args.metric
        if args.deterministic: set_seeds()
    
        if not args.dont_save:
            ## Log
            print('Writing to the experiment directory: {}'.format(self._hp.exp_path))
            if not os.path.exists(self._hp.exp_path):
                os.makedirs(self._hp.exp_path)
            save_cmd(self._hp.exp_path)
            save_git(self._hp.exp_path)
            save_config(conf_path, os.path.join(self._hp.exp_path, "conf_" + datetime_str() + ".py"))
    
        self.use_cuda = torch.cuda.is_available() and not global_params.debug
        self.device = torch.device('cuda') if self.use_cuda else torch.device('cpu')
    
        ## Buld dataset, model. logger, etc.
    
        if not self.args.dont_save:
            writer = SummaryWriter(log_dir, max_queue=1, flush_secs=1)
        else:
            writer = None
        # TODO clean up param passing
        model_conf['batch_size'] = self._hp.batch_size if not torch.cuda.is_available() else int(
            self._hp.batch_size / torch.cuda.device_count())
        # copy over data specs, but cannot copy list into hparams
        model_conf.update({i: data_conf.dataset_spec[i] for i in data_conf.dataset_spec
                           if not isinstance(data_conf.dataset_spec[i], list)})
        model_conf['device'] = self.device.type
    
        def build_phase(logger, model, phase, n_repeat=1, dataset_size=-1):
            if not self.args.dont_save:
                logger = logger(log_dir, self._hp, max_seq_len=model_conf['max_seq_len'], summary_writer=writer)
                                # fps=data_conf.fps)
            else:
                logger = None
            model = model(model_conf, logger)
            if torch.cuda.device_count() > 1:
                print("\nUsing {} GPUs!\n".format(torch.cuda.device_count()))
                model = DataParallelWrapper(model)
            model = model.to(self.device)
            model.device = self.device
            loader = self.get_dataset(args, model, data_conf, phase, n_repeat, dataset_size)
        
            return logger, model, loader

        self.logger, self.model, self.train_loader = build_phase(self._hp.logger, self._hp.model, 'train',
                                                                 n_repeat=self._hp.epoch_cycles_train)
    
        if self.run_testmetrics:
            phase = 'test'
        else:
            phase = 'val'
        self.logger_test, _, self.val_loader = \
            build_phase(self._hp.logger_test, self._hp.model_test, phase, dataset_size=args.val_data_size)
    
        self.optimizer = self.get_optimizer_class()(filter(lambda p: p.requires_grad, self.model.parameters()),
                                                    lr=self._hp.lr)
        self._hp.mpar = self.model._hp
    
        if hasattr(self.model, 'dense_rec') and not self.args.skip_top100_val:
            self.evaluator = Evaluator(self.model, self.log_dir, self._hp, self.run_testmetrics, self.logger_test)
        else:
            self.evaluator = None
    
        if hasattr(conf_module, 'control_config'):
            control_conf = conf_module.control_config
            control_conf.policy.model = self.model
            control_conf.data_save_dir = self._hp.exp_path + '/control'
            self.control_manager = ControlManager(args_in=['None'], hyperparams=control_conf)
        else:
            self.control_manager = None
    
        self.training_context = autograd.detect_anomaly if args.detect_anomaly else dummy_context
        self.hooks = []
        self.hooks.append(LossSpikeHook('sg_img_mse_train'))
        self.hooks.append(NanGradHook(self))
        # SVG has none gradients for the tree part of the network
        if self.model._hp.dense_rec_type != 'svg': self.hooks.append(NoneGradHook(self))
    
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

    def get_dataset(self, args, model, data_conf, phase, n_repeat, dataset_size=-1):
        if args.feed_random_data:
            from gcp.datasets.data_generator import RandomVideoDataset
            dataset_class = RandomVideoDataset
        elif 'dataset_class' in data_conf.dataset_spec:
            dataset_class = data_conf.dataset_spec.dataset_class
            vis_utils.PARAMS.visualize = dataset_class.visualize
        elif hasattr(data_conf, 'dataset_conf') and 'dataset_class' in data_conf.dataset_conf:
            dataset_class = data_conf.dataset_conf.dataset_class
            vis_utils.PARAMS.visualize = dataset_class.visualize
        else:
            dataset_class = FolderSplitVarLenVideoDataset
    
        loader = dataset_class(get_dataset_path(self._hp.dataset_name), model._hp, data_conf,
                               phase=phase, shuffle=not self.run_testmetrics, dataset_size=dataset_size). \
            get_data_loader(self._hp.batch_size, n_repeat)
    
        return loader

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
                                           strict=self.args.strict_weight_loading)
        self.model.to(self.model.device)
        return start_epoch

    def get_configs(self):
        self.args = args = self.get_trainer_args()
        exp_dir = get_exp_dir()
        conf_path = get_config_path(args.path)
        print('loading from the config file {}'.format(conf_path))
        conf_module = imp.load_source('conf', conf_path)
        conf = conf_module.configuration
        model_conf = conf_module.model_config
    
        if args.gpu != -1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
        return args, conf_module, conf, model_conf, exp_dir, conf_path

    def get_data_config(self, conf_module):
        # get default data config
        path = os.path.join(get_dataset_path(conf_module.configuration['dataset_name']), 'dataset_spec.py')
        data_conf_file = imp.load_source('dataset_spec', path)
        data_conf = AttrDict()
        data_conf.dataset_spec = AttrDict(data_conf_file.dataset_spec)

        # update with custom params if available
        try:
            update_data_conf = conf_module.data_config
        except AttributeError:
            pass
        for key in update_data_conf:
            if key == "dataset_spec":
                data_conf.dataset_spec.update(update_data_conf.dataset_spec)
            else:
                data_conf[key] = update_data_conf[key]

        if not 'fps' in data_conf:
            data_conf.fps = 4
        return data_conf

    def get_optimizer_class(self):
        optim = self._hp.optimizer
        if optim == 'adam':
            get_optim = partial(get_clipped_optimizer, optimizer_type=Adam, betas=(self._hp.adam_beta, 0.999))
        elif optim == 'radam':
            get_optim = partial(get_clipped_optimizer, optimizer_type=RAdam, betas=(self._hp.adam_beta, 0.999))
        elif optim == 'rmsprop':
            get_optim = partial(get_clipped_optimizer, optimizer_type=RMSprop, momentum=self._hp.momentum)
        elif optim == 'sgd':
            get_optim = partial(get_clipped_optimizer, optimizer_type=SGD, momentum=self._hp.momentum)
        else:
            raise ValueError("Optimizer '{}' not supported!".format(optim))
        return partial(get_optim, gradient_clip=self._hp.gradient_clip)

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
        
            self.logger_test, self.model, self.val_loader = \
                rebuild_phase(self._hp.logger_test, self._hp.model_test, "val", dataset_size=args.val_data_size)
            self.evaluator = Evaluator(self.model, log_dir, self._hp, self.run_testmetrics, self.logger_test)
            if args.resume:
                self.resume(args.resume)
            else:
                self.resume("latest")
            self.val()
        print("...Done!")

    def get_trainer_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--path", help="path to the config file directory")
    
        # Folder settings
        parser.add_argument("--prefix", help="experiment prefix, if given creates subfolder in experiment directory")
        parser.add_argument('--new_dir', default=False, type=int, help='If True, concat datetime string to exp_dir.')
        parser.add_argument('--dont_save', default=False, type=int,
                            help="if True, nothing is saved to disk. Note: this doesn't work")  # TODO this doesn't work
    
        parser.add_argument("--visualize", default='', help="path to model file to visualize")  # TODO what uses this?
    
        # Running protocol
        parser.add_argument('--resume', default='latest', type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')
        parser.add_argument('--train', default=True, type=int,
                            help='if False, will run one validation epoch')
        parser.add_argument('--test_control', default=True, type=int,
                            help="if False, control isn't run at validation time")
        parser.add_argument('--test_prediction', default=True, type=int,
                            help="if False, prediction isn't run at validation time")
        parser.add_argument('--skip_first_val', default=False, type=int,
                            help='if True, will skip the first validation epoch')
        parser.add_argument('--skip_top100_val', default=False, type=int,
                            help="if True, skips top of 100 eval")
        parser.add_argument('--metric', default=False, type=int,
                            help='if True, run test metrics')
        parser.add_argument('--val_sweep', default=False, type=int,
                            help='if True, runs validation on all existing model checkpoints')
        parser.add_argument('--dataset_val_sweep', default=False, type=int,
                            help='if True, runs validation on a given collection of datasets')
    
        # Misc
        parser.add_argument('--gpu', default=-1, type=int,
                            help='will set CUDA_VISIBLE_DEVICES to selected value')
        parser.add_argument('--strict_weight_loading', default=True, type=int,
                            help='if True, uses strict weight loading function')
        parser.add_argument('--deterministic', default=False, type=int,
                            help='if True, sets fixed seeds for torch and numpy')
        parser.add_argument('--imepoch', default=4, type=int,
                            help='number of image loggings per epoch')
        parser.add_argument('--val_data_size', default=-1, type=int,
                            help='number of sequences in the validation set. If -1, the full dataset is used')
        parser.add_argument('--log_images', default=True, type=int,
                            help='')
        parser.add_argument('--log_outputs_interval', default=10, type=int,
                            help='')
    
        # Debug
        parser.add_argument('--detect_anomaly', default=False, type=int,
                            help='if True, uses autograd.detect_anomaly()')
        parser.add_argument('--feed_random_data', default=False, type=int,
                            help='if True, we feed random data to the model to test its performance')
        parser.add_argument('--train_loop_pdb', default=False, type=int,
                            help='if True, opens a pdb into training loop')
        parser.add_argument('--debug', default=False, type=int,
                            help='if True, runs in debug mode')
        parser.add_argument('--verbose_timing', default=False, type=int,
                            help='if True, prints additional time measurements.')
        return parser.parse_args()

    def _default_hparams(self):
        # put new parameters in here:
        default_dict = {
            'model': None,
            'model_test': None,
            'logger': None,
            'logger_test': None,
            'dataset_name': None,  # directory where dataset is in
            'batch_size': 64,
            'mpar': None,  # model parameters
            'exp_path': None,  # Path to the folder with experiments
            'num_epochs': 200,
            'epoch_cycles_train': 1,
            'mujoco_xml': None,
            'optimizer': 'radam',  # supported: 'radam', 'adam', 'rmsprop', 'sgd'
            'lr': None,
            'gradient_clip': None,
            'momentum': 0,  # momentum in RMSProp / SGD optimizer
            'adam_beta': 0.9,  # beta1 param in Adam
            'metric_pruning_scheme': 'dtw',
            'top_of_100_eval': True,
            'n_rooms': None,
        }
        # add new params to parent params
        parent_params = HParams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params
