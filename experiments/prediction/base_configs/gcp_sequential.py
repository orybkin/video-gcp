from gcp.models.svg import SVGModel
from gcp.rec_planner_utils.logger import HierarchyLogger

configuration = {
    'model': SVGModel,
    'model_test': SVGModel,
    'logger': HierarchyLogger,
    'logger_test': HierarchyLogger,
}

model_config = {
    'one_step_planner': 'continuous',
    'dense_rec_type': 'svg',
    'hierarchy_levels': 0,
    'add_weighted_pixel_copy': True,
}