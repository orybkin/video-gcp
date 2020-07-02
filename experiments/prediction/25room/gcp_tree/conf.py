import os

from blox import AttrDict
from gcp.datasets.data_loader import MazeTopRenderedGlobalSplitVarLenVideoDataset
from gcp.planning.cem.cost_fcn import EuclideanPathLength

current_dir = os.path.dirname(os.path.realpath(__file__))
from experiments.prediction.base_configs import gcp_tree as base_conf

configuration = AttrDict(base_conf.configuration)
configuration.update({
    'dataset_name': 'nav_25rooms',
    'batch_size': 16,
    'lr': 2e-4,
    'epoch_cycles_train': 2,
    'n_rooms': 25,
    'metric_pruning_scheme': 'basic',
})

model_config = AttrDict(base_conf.model_config)
model_config.update({
    'untied_layers': True,
    'hierarchy_levels': 8,
    'ngf': 16,
    'nz_mid_lstm': 512,
    'n_lstm_layers': 3,
    'nz_mid': 128,
    'nz_enc': 128,
    'nz_vae': 256,
    'regress_length': True,
    'attach_state_regressor': True,
    'attach_cost_mdl': True,
    'cost_mdl_params': AttrDict(
        cost_fcn=EuclideanPathLength,
    ),
    'attach_inv_mdl': True,
    'inv_mdl_params': AttrDict(
        n_actions=2,
        use_convs=False,
        build_encoder=False,
    ),
    'decoder_distribution': 'discrete_logistic_mixture',
})
model_config.pop("add_weighted_pixel_copy")