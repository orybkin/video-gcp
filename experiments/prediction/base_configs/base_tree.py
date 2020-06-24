from blox import AttrDict
from gcp.models.base_hierarchical_planner import HierarchicalPlanner, HierarchicalPlannerTest
from gcp.rec_planner_utils.logger import HierarchyLogger

configuration = {
    'model': HierarchicalPlanner,
    'model_test': HierarchicalPlannerTest,
    'logger': HierarchyLogger,
    'logger_test': HierarchyLogger,
}
configuration = AttrDict(configuration)

model_config = {
    'one_step_planner': 'sh_pred',
    'hierarchy_levels': 7,
    'matcher': 'loss',
    'seq_enc': 'conv',
    'tree_lstm': 'split_linear',
    'lstm_init': 'mlp',
    'add_weighted_pixel_copy': True,
    'dense_rec_type': 'node_prob',
}


