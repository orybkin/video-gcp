from gcp.models.hedge import SHPredModule
import torch.nn as nn


""" This module contains the hierarchical model in which every layer is predicted by a separately trained network. """


class UntiedLayersInpainter(nn.Module):
    def __init__(self, hp, decoder):
        super().__init__()
        self._hp = hp
        self.one_step_planners = nn.ModuleList([SHPredModule(hp, decoder) for i in range(self._hp.hierarchy_levels)])

    def produce_subgoal(self, *args, depth, **kwargs):
        return self.one_step_planners[self._hp.hierarchy_levels - depth].produce_subgoal(*args, **kwargs)
    
    def __getattr__(self, item):
        if item in self._modules.keys():
            return super().__getattr__(item)
            
        return getattr(self.one_step_planners[0], item)
