import matplotlib;
from gcp.prediction.utils.utils import datetime_str, make_path, set_seeds, get_dataset_path, download_data

matplotlib.use('Agg')
import torch
import argparse
import os
from shutil import copy
import imp
import importlib
from tensorflow.contrib.training import HParams
from tensorboardX import SummaryWriter
from torch import autograd
from torch.optim import Adam, RMSprop, SGD
from functools import partial

from gcp.datasets.data_loader import FolderSplitVarLenVideoDataset
from gcp.prediction.training.checkpoint_handler import save_cmd, save_git, get_config_path
from blox.utils import dummy_context
from blox import AttrDict
from blox.torch.training import LossSpikeHook, NanGradHook, NoneGradHook, DataParallelWrapper, \
    get_clipped_optimizer
from gcp.evaluation.compute_metrics import Evaluator
from gcp.prediction.training.base_trainer import BaseTrainer
from gcp.prediction import global_params
from gcp.prediction.utils import visualization
from blox.torch.radam import RAdam



class GCPBuilder(BaseTrainer):
    """ This class constructs the GCP model, dataset and the optimizers """

    def __init__(self):
        self.batch_idx = 0
    
        ## Set up params
        cmd_args, model_conf, conf_path, data_conf = self.get_configs()
    
        ## Set up logging
        self.log_dir = log_dir = os.path.join(self._hp.exp_path, 'events')
        print('using log dir: ', log_dir)
        if not cmd_args.dont_save:
            # Log
            print('Writing to the experiment directory: {}'.format(self._hp.exp_path))
            if not os.path.exists(self._hp.exp_path):
                os.makedirs(self._hp.exp_path)
            save_cmd(self._hp.exp_path)
            save_git(self._hp.exp_path)
            # Copy config file
            copy(conf_path, os.path.join(self._hp.exp_path, "conf_" + datetime_str() + ".py"))
            writer = SummaryWriter(log_dir, max_queue=1, flush_secs=1)
            self.logger = self._hp.logger(log_dir, self._hp, max_seq_len=data_conf.dataset_spec.max_seq_len,
                                          summary_writer=writer)
        else:
            self.logger = None

        ## Set up CUDA
        self.use_cuda = torch.cuda.is_available() and not global_params.debug
        self.device = torch.device('cuda') if self.use_cuda else torch.device('cpu')
        if cmd_args.gpu != -1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cmd_args.gpu)
        if cmd_args.deterministic:
            set_seeds()

        ## Set up model conf
        # copy over data specs, but cannot copy list into hparams
        model_conf.update({k: v for k, v in data_conf.dataset_spec.items() if not isinstance(v, list)})
        model_conf['device'] = self.device.type
        model_conf['batch_size'] = self._hp.batch_size
        if self.use_cuda:
            model_conf['batch_size'] = int(self._hp.batch_size / torch.cuda.device_count())

        ## Build model
        model = self._hp.model(model_conf, self.logger)
        if torch.cuda.device_count() > 1:
            print("\nUsing {} GPUs!\n".format(torch.cuda.device_count()))
            model = DataParallelWrapper(model)
        model.device = self.device
        self.model = model = model.to(self.device)
    
        ## Build data loading
        self.train_loader = self.get_dataset(model, data_conf, 'train', self._hp.epoch_cycles_train, -1)
        phase = 'test' if self.cmd_args.metric else 'val'
        self.val_loader = self.get_dataset(model, data_conf, phase, self._hp.epoch_cycles_train, cmd_args.val_data_size)
    
        ## Build optimizer
        self.optimizer = self.get_optimizer_class()(filter(lambda p: p.requires_grad, self.model.parameters()),
                                                    lr=self._hp.lr)
    
        ## Build evaluator
        if hasattr(self.model, 'dense_rec') and not self.cmd_args.skip_top100_val:
            self.evaluator = Evaluator(self.model, self.log_dir, self._hp, self.cmd_args.metric, self.logger)
        else:
            self.evaluator = None
    
        ## Set up training options: context, hooks
        self.training_context = autograd.detect_anomaly if cmd_args.detect_anomaly else dummy_context
        self.hooks = []
        self.hooks.append(LossSpikeHook('sg_img_mse_train'))
        self.hooks.append(NanGradHook(self))
        # SVG has none gradients for the tree part of the network
        if self.model._hp.dense_rec_type != 'svg': self.hooks.append(NoneGradHook(self))

    def get_dataset(self, model, data_conf, phase, n_repeat, dataset_size=-1):
        if self.cmd_args.feed_random_data:
            from gcp.datasets.data_generator import RandomVideoDataset
            dataset_class = RandomVideoDataset
        elif 'dataset_class' in data_conf.dataset_spec:
            dataset_class = data_conf.dataset_spec.dataset_class
            visualization.PARAMS.visualize = dataset_class.visualize
        elif hasattr(data_conf, 'dataset_conf') and 'dataset_class' in data_conf.dataset_conf:
            dataset_class = data_conf.dataset_conf.dataset_class
            visualization.PARAMS.visualize = dataset_class.visualize
        else:
            dataset_class = FolderSplitVarLenVideoDataset
    
        loader = dataset_class(get_dataset_path(self._hp.dataset_name), model._hp, data_conf,
                               phase=phase, shuffle=not self.cmd_args.metric, dataset_size=dataset_size). \
            get_data_loader(self._hp.batch_size, n_repeat)
    
        return loader

    def get_configs(self):
        # Cmd arguments
        self.cmd_args = cmd_args = self.get_cmd_args()
        exp_dir = os.environ['GCP_EXP_DIR']

        # Config file
        conf_path = get_config_path(cmd_args.path)
        print('loading from the config file {}'.format(conf_path))
        conf_module = imp.load_source('conf', conf_path)

        # Trainer config
        trainer_conf = conf_module.configuration
        self._hp = self._default_hparams()
        self.override_defaults(trainer_conf)  # override defaults with config file
        self._hp.set_hparam('exp_path', make_path(exp_dir, cmd_args.path, cmd_args.prefix, cmd_args.new_dir))

        # Model config
        model_conf = conf_module.model_config

        # Data config
        download_data(trainer_conf.dataset_name)
        data_conf = self.get_data_config(conf_module)
    
        return cmd_args, model_conf, conf_path, data_conf

    def get_data_config(self, conf_module):
        # get default data config
        
        path = os.path.join(get_dataset_path(conf_module.configuration['dataset_name']), 'dataset_spec.py')
        data_conf_file = imp.load_source('dataset_spec', path)
        data_conf = AttrDict()
        data_conf.dataset_spec = AttrDict(data_conf_file.dataset_spec)

        # update with custom params if available
        update_data_conf = {}
        if hasattr(conf_module, 'data_config'):
            update_data_conf = conf_module.data_config
        elif conf_module.configuration.dataset_name is not None:
            update_data_conf = importlib.import_module('gcp.datasets.configs.' + conf_module.configuration.dataset_name).config
            
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

    def get_cmd_args(self):
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
            'logger': None,
            'dataset_name': None,  # directory where dataset is in
            'batch_size': 64,
            'exp_path': None,  # Path to the folder with experiments
            'num_epochs': 200,
            'epoch_cycles_train': 1,
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
