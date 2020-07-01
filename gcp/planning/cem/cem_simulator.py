import torch
import numpy as np

from blox import AttrDict


class GCPSimulator:
    """Implements simple simulator interface for GCP models."""
    def __init__(self, model, append_latent):
        self._model = model
        self._append_latent = append_latent  # if True, appends the latent to the state rollout
        self._logs = []

    def rollout(self, state, goal_state, samples, rollout_len, prune=False):
        """Performs one model rollout."""
        # prepare inputs
        batch_size = samples.shape[0]
        state, goal_state = state.repeat(batch_size, 0), goal_state.repeat(batch_size, 0)
        input_dict = AttrDict(I_0=torch.tensor(state, device=self._model.device, dtype=torch.float32),
                              I_g=torch.tensor(goal_state, device=self._model.device, dtype=torch.float32),
                              start_ind=torch.tensor(np.zeros((batch_size,)), device=self._model.device).long(),
                              end_ind=torch.tensor(np.ones((batch_size,)) * (rollout_len - 1),
                                                   device=self._model.device).long(),
                              z=torch.tensor(samples, device=self._model.device, dtype=torch.float32))
        input_dict = self._postprocess_inputs(input_dict)

        # perform rollout, collect outputs
        outputs = AttrDict()
        with self._model.val_mode():
            model_output = self._model(input_dict)
            end_ind = torch.max(model_output.end_ind, torch.ones_like(model_output.end_ind))
        # self._logs.append(model_output)

        if prune:
            outputs.predictions = self._list2np(model_output.pruned_prediction)
        else:
            outputs.predictions = self._list2np(self._get_state_rollouts(input_dict, model_output, end_ind))

        outputs.actions = self._list2np(self._cap_to_length(model_output.actions, end_ind))
        outputs.states = self._list2np(self._cap_to_length(model_output.regressed_state, end_ind))
        outputs.latents = self._list2np(self._cap_to_length(input_dict.model_enc_seq, end_ind))

        return outputs

    def _postprocess_inputs(self, input_dict):
        return input_dict

    def _get_state_rollouts(self, input_dict, model_output, end_ind):
        batch_size = model_output.end_ind.shape[0]
        state_plans = []
        for i in range(batch_size):
            seq_len = end_ind[i] + 1
            out = self._model.dense_rec.get_sample_with_len(i, seq_len, model_output, input_dict, 'basic')
            rollout = out[0].reshape(seq_len, -1)
            if self._append_latent:
                name = 'encodings' if model_output.dense_rec else 'e_g_prime'
                latent_rollout = self._model.dense_rec.get_sample_with_len(
                    i, seq_len, model_output, input_dict, 'basic', name=name)[0].reshape(seq_len, -1)
                rollout = torch.cat((rollout, latent_rollout), dim=-1)
            state_plans.append(rollout)
        return state_plans

    @staticmethod
    def _cap_to_length(vals, end_inds):
        assert vals.shape[0] == end_inds.shape[0]
        return [val[:end_ind + 1] for val, end_ind in zip(vals, end_inds)]

    @staticmethod
    def _list2np(list):
        return [elem.data.cpu().numpy() for elem in list]

    def dump_logs(self, dump_file='rollout_dump.pkl'):
        import pickle
        with open(dump_file, 'wb') as F:
            pickle.dump(self._logs, F)
        self._logs = []


class GCPImageSimulator(GCPSimulator):
    def _postprocess_inputs(self, input_dict):
        input_dict = super()._postprocess_inputs(input_dict)
        input_dict.z = input_dict.z[..., None, None]
        input_dict.I_0 = self._env2planner(input_dict.I_0)
        input_dict.I_g = self._env2planner(input_dict.I_g)
        return input_dict

    @staticmethod
    def _env2planner(img):
        """Converts images to the [-1...1] range of the hierarchical planner."""
        if img.max() > 1.0:
            img = img / 255.0
        if len(img.shape) == 5:
            img = img[0]
        if len(img.shape) == 4:
            img = img.permute(0, 3, 1, 2)
        return img * 2 - 1.0


class ActCondGCPImageSimulator(GCPImageSimulator):
    def _postprocess_inputs(self, input_dict):
        input_dict = super()._postprocess_inputs(input_dict)
        input_dict.actions = input_dict.pop('z')[..., 0, 0]     # input (action) samples are stored in z by default
        input_dict.pad_mask = torch.ones(input_dict.actions.shape[:2], device=input_dict.actions.device)
        return input_dict



