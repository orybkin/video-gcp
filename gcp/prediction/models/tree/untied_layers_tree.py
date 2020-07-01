import torch.nn as nn

from gcp.prediction.models.tree.tree_module import TreeModule

""" This module contains the hierarchical model in which every layer is predicted by a separately trained network. """


class UntiedLayersTree(nn.Module):
    def __init__(self, hp, decoder):
        super().__init__()
        self._hp = hp
        self.tree_modules = nn.ModuleList([TreeModule(hp, decoder) for i in range(self._hp.hierarchy_levels)])

    def produce_subgoal(self, *args, depth, **kwargs):
        return self.tree_modules[self._hp.hierarchy_levels - depth].produce_subgoal(*args, **kwargs)
    
    def __getattr__(self, item):
        if item in self._modules.keys():
            return super().__getattr__(item)
            
        return getattr(self.tree_modules[0], item)
