from gcp.prediction.models.sequential import SequentialModel
from gcp.prediction.utils.logger import HierarchyLogger

configuration = {
    'model': SequentialModel,
    'logger': HierarchyLogger,
}

model_config = {
    'one_step_planner': 'continuous',
    'dense_rec_type': 'svg',
    'hierarchy_levels': 0,
    'add_weighted_pixel_copy': True,
}