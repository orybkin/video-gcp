import os

from blox import AttrDict

current_dir = os.path.dirname(os.path.realpath(__file__))
from experiments.prediction.base_configs import gcp_sequential as base_conf

configuration = AttrDict(base_conf.configuration)

model_config = base_conf.model_config
model_config.update({
    'action_conditioned_pred': True,
    'non_goal_conditioned': True,
    'nz_vae': 0,
    'var_inf': 'deterministic',
})
