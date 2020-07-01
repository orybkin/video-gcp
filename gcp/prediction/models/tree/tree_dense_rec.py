from blox import AttrDict
from blox.torch.modules import DummyModule
from gcp.evaluation.evaluation_matching import DTWEvalBinding, BalancedPrunedDTWBinding, BalancedEvalBinding


class TreeDenseRec(DummyModule):
    def __init__(self, hp, *_, decoder, **__):
        super().__init__()
        self._hp = hp
        self.eval_binding = None
        self.decoder = decoder

    def get_sample_with_len(self, i_ex, length, outputs, inputs, pruning_scheme, name=None):
        """Perform evaluation matching, return dense sequence of specified length."""
        if self.eval_binding is None:
            self.eval_binding = self._get_eval_binding(pruning_scheme)
        return self.eval_binding(outputs, inputs, length, i_ex, name)
    
    def get_all_samples_with_len(self, length, outputs, inputs, pruning_scheme, name=None):
        """Perform evaluation matching, return dense sequence of specified length."""
        if self.eval_binding is None:
            self.eval_binding = self._get_eval_binding(pruning_scheme)
            
        if hasattr(self.eval_binding, 'get_all_samples'):
            return self.eval_binding.get_all_samples(outputs, inputs, length, name)
        else:
            return [self.eval_binding(outputs, inputs, length, i_ex, name) for i_ex in range(outputs.end_ind.shape[0])]
 
    def _get_eval_binding(self, pruning_scheme):
        if pruning_scheme == 'dtw':
            return DTWEvalBinding(self._hp)
        if pruning_scheme == 'pruned_dtw':
            assert self._hp.matching_type == 'balanced'
            return BalancedPrunedDTWBinding(self._hp)
        if pruning_scheme == 'basic':
            assert self._hp.matching_type == 'balanced'
            return BalancedEvalBinding(self._hp)
        else:
            raise ValueError("Eval pruning scheme {} not currently supported!".format(pruning_scheme))
        
    def forward(self, tree, inputs):
        decoded_seq = self.decoder.decode_seq(inputs, tree.bf.e_g_prime)
        tree.set_attr_bf(**decoded_seq)
        return AttrDict()
