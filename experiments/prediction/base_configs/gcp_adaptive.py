from blox import AttrDict
from experiments.prediction.base_configs import base_tree as base_conf

configuration = AttrDict(base_conf.configuration)

model_config = AttrDict(base_conf.model_config)
model_config.update({
    'matching_type': 'dtw_image',
    'learn_matching_temp': False,
})
