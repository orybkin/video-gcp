from blox import AttrDict
from gcp.prediction.models.tree.tree import TreeModel
from gcp.prediction.utils.logger import HierarchyLogger

configuration = {
    'model': TreeModel,
    'logger': HierarchyLogger,
}
configuration = AttrDict(configuration)

model_config = {
    'one_step_planner': 'sh_pred',
    'hierarchy_levels': 7,
    'binding': 'loss',
    'seq_enc': 'conv',
    'tree_lstm': 'split_linear',
    'lstm_init': 'mlp',
    'add_weighted_pixel_copy': True,
    'dense_rec_type': 'node_prob',
}

