import numpy as np
import torch

from blox import AttrDict, optional
from blox.torch.losses import L2Loss
from gcp.rec_planner_utils.matching import BaseMatcher


class AttentiveMatcherWithIntegration(BaseMatcher):
    """ A matching scheme that sort of works """

    def __init__(self, hp, type, temp, moment_matching=False):
        raise NotImplementedError("Deprecated! Not updated to layer-wise tree computation!")
        super().__init__(hp)
        self.type = type
        self.temp = temp
        self.moment_matching = moment_matching
        self.key = 'c_n_prime'  # The key that needs to be collected from the tree and passed to the loss

        self.pre_dists_sum = None

    def __call__(self, inputs, subgoal, left_parent, right_parent):
        out = AttrDict()
        out.c_n = self.attentive_matching(inputs, subgoal)
        out.c_n_prime, out.cdf, out.p_n = self.propagate_matching(subgoal, left_parent, right_parent, out.c_n)
        out.ind = torch.argmax(out.c_n_prime, dim=1)

        return out

    def attentive_matching(self, inputs, subgoal):
        """Computes TAP-style matching distribution based on image MSE distance."""
        if self.type == 'image':
            assert subgoal.images is not None  # need subgoal image decodings
            target, estimate = inputs.demo_seq, subgoal.images
        elif self.type == 'latent':
            target, estimate = inputs.enc_demo_seq, subgoal.e_g_prime_preact
        else:
            raise ValueError("Matching type {} not supported!".format(self.type))

        batch, time = inputs.demo_seq.shape[:2]
        mse_diff = torch.nn.MSELoss(reduction='none')(target, estimate[:, None])
        mse_diff = mse_diff.view(batch, time, -1).mean(2)
        mse_diff[(1 - inputs.pad_mask).type(torch.ByteTensor)] = np.inf  # mask diff to non-sequence padding frames
        matching_scores = torch.softmax(-self.temp * mse_diff, dim=1)

        if self.moment_matching:
            # fit a Gaussian to the matching distribution
            time_steps = torch.arange(time, device=matching_scores.device).float()[None, :].expand(batch, -1)
            expect = torch.sum(time_steps * matching_scores, dim=1)
            std_dev = torch.sum((time_steps - expect[:, None])**2 * matching_scores, dim=1).sqrt()
            # TODO fix the Gaussian computation to properly sum to 1
            matching_scores = torch.distributions.Normal(expect, std_dev). \
                log_prob(time_steps.transpose(0, 1)).exp().transpose(0, 1)  # transpose for Normal requirements
            matching_scores = matching_scores * inputs.pad_mask     # mask matchings after end of sequence

        return matching_scores

    def propagate_matching(self, node, left_parent, right_parent, c_n_prime):
        """Propagates the parent matches to the current subgoal match."""
        if left_parent is not None:
            l_cdf = left_parent.cdf
            left_pad = torch.zeros(l_cdf.shape[0], 1,
                                   device=l_cdf.device)  # switch left parent's cdf one to right to get proper probability
            l_cdf_prime = torch.cat((left_pad, l_cdf[:, 1:]), dim=1)
            c_n_prime = c_n_prime * l_cdf_prime

        if right_parent is not None:
            r_cdf = right_parent.cdf
            c_n_prime = c_n_prime * (
            r_cdf[..., -1][..., None] - r_cdf)  # max(r_cdf) - r_cdf, because cdf is not normalized
        cdf = torch.cumsum(c_n_prime, dim=1)
        p_n = cdf[..., -1]  # probability of node existence

        return c_n_prime, cdf, p_n

    @optional()
    def _norm_match_dists(self, match_dists, pad_mask):
        add_constant = self._hp.minent > 0.0
        if add_constant:
            match_dists = match_dists + self._hp.minent

        dists_sum = torch.sum(match_dists, dim=1, keepdim=True)
        if not add_constant:
            dists_sum = torch.clamp(dists_sum, 1e-12)

        norm_match_dists = (match_dists / dists_sum) * pad_mask[:, None, :match_dists.shape[-1]]
        return norm_match_dists

    def get_w(self, match_dists, pad_mask, log=False):
        """ Transforms c' into w """

        if log:
            self.pre_dists_sum = torch.sum(match_dists, dim=1)[pad_mask.byte()]
        gt_match_dists = self._norm_match_dists(match_dists, pad_mask, yes=self._hp.norm_match_dists)

        return gt_match_dists

    def log_outputs_stateful(self, step, log_images, phase, logger):
        if self.pre_dists_sum is None: return   # need to call 'forward' at least once with log=True before!
        logger.log_scalar(self.pre_dists_sum.min(), 'matching_dists_w/min', step, phase)
        logger.log_scalar((self.pre_dists_sum == 0).sum(), 'matching_dists_w/n_zero', step, phase)
        logger.log_scalar((self.pre_dists_sum < self._hp.minent).sum(), 'matching_dists_w/n_highent', step, phase)

    def loss(self):
        est = self.pre_dists_sum
        explaining = L2Loss(self._hp.hack_explaining_loss)(est.float().mean(), 1)
        return AttrDict(explaining=explaining)