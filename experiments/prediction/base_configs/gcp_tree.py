from blox import AttrDict
from experiments.prediction.base_configs import base_tree as base_conf

configuration = AttrDict(base_conf.configuration)
configuration.metric_pruning_scheme = 'pruned_dtw'

model_config = AttrDict(base_conf.model_config)
model_config.update({
    'matching_type': 'balanced',
    'forced_attention': True,
})
