import os
from contextlib import contextmanager

import numpy as np
import torch
from skimage.io import imsave

from blox import AttrDict
from blox.basic_types import dict_concat
from blox.tensor.ops import batchwise_index
from blox.torch.evaluation import ssim, psnr, mse
from blox.utils import timed
from gcp.prediction.utils.visualization import plot_pruned_tree, make_gif


def make_image_strips(input, gen_seq, phase, outdir, ind):
    """
    :param input:
    :param gen_seq:   t, channel, r, c
    :param ind: batch index to make images for
    :param phase:
    :param outdir:
    :return:
    """

    gen_seq = gen_seq.detach().cpu().numpy()
    gen_seq = np.split(gen_seq, gen_seq.shape[0], axis=0)
    gen_seq = [l.squeeze() for l in gen_seq]
    gen_seq = [np.transpose(item,(1,2,0)) for item in gen_seq]

    input = input.detach().cpu().numpy()
    input = np.split(input, input.shape[0], axis=0)
    input = [l.squeeze() for l in input]
    input = [np.transpose(item,(1,2,0)) for item in input]

    input = np.concatenate(input, axis=1)
    gen_seq = np.concatenate(gen_seq, axis=1)

    input = (input + 1)/2
    gen_seq = (gen_seq + 1)/2

    out = np.concatenate([input, gen_seq], axis=0)

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    imsave(outdir + '/outfile{}_{}.png'.format(ind, phase), out)


class Evaluator:
    """Performs evaluation of metrics etc."""
    N_PLOTTED_ELEMENTS = 5
    LOWER_IS_BETTER_METRICS = ['mse']
    HIGHER_IS_BETTER_METRICS = ['psnr', 'ssim']

    def __init__(self, model, logdir, hp, log_to_file, tb_logger, top_comp_metric='mse'):
        self._logdir = logdir + '/metrics'
        self._logger = FileEvalLogger(self._logdir) if log_to_file else TBEvalLogger(logdir, tb_logger)
        self._hp = hp
        self._pruning_scheme = hp.metric_pruning_scheme
        self._dense_rec_module = model.dense_rec
        self.use_images = model._hp.use_convs
        self._top_of_100 = hp.top_of_100_eval
        self._top_of = 100
        self._top_comp_metric = top_comp_metric
        if not os.path.exists(self._logdir): os.makedirs(self._logdir)
        self.evaluation_buffer = None
        self.full_evaluation = None
        self.dummy_env = None

    def reset(self):
        self.evaluation_buffer = None
        self.full_evaluation = None

    def _erase_eval_buffer(self):
        def get_init_array(val):
            n_eval_samples = self._top_of if self._top_of_100 else 1
            return val * np.ones((self._hp.batch_size, n_eval_samples))
        self.evaluation_buffer = AttrDict(ssim=get_init_array(0.),
                                          psnr=get_init_array(0.),
                                          mse=get_init_array(np.inf),
                                          gen_images=np.empty(self._hp.batch_size, dtype=np.object),
                                          rand_seqs=np.empty(self._hp.batch_size, dtype=np.object))
        for b in range(self._hp.batch_size):
            self.evaluation_buffer.rand_seqs[b] = []
        if not self.use_images:
            self.evaluation_buffer.pop('ssim')
            self.evaluation_buffer.pop('psnr')

    def eval_single(self, inputs, outputs, sample_n=0):
        input_images = inputs.traj_seq
        bsize = input_images.shape[0]
        store_states = "traj_seq_states" in inputs and (inputs.traj_seq_states.shape[-1] == 2 or
                                                        inputs.traj_seq_states.shape[-1] == 5)
        # TODO paralellize DTW
        for b in range(bsize):
            input_seq = input_images[b, :inputs.end_ind[b]+1]
            input_len = input_seq.shape[0]
            gen_seq, matching_output = self._dense_rec_module.get_sample_with_len(b, input_len, outputs, inputs, self._pruning_scheme)
            input_seq, gen_seq = input_seq[1:-1], gen_seq[1:-1]     # crop first and last frame for eval (conditioning frames)
            state_seq = inputs.traj_seq_states[b, :input_len] if store_states else None

            full_gen_seq, gen_seq = self.compute_metrics(b, gen_seq, input_seq, outputs, sample_n)

            if self._is_better(self.evaluation_buffer[self._top_comp_metric][b, sample_n],
                               self.evaluation_buffer[self._top_comp_metric][b]):
                # log visualization results for the best sample only, replace if better
                self.evaluation_buffer.gen_images[b] = AttrDict(gt_seq=input_images.cpu().numpy()[b],
                                                                gen_images=gen_seq,
                                                                full_gen_seq=full_gen_seq,
                                                                matching_outputs=matching_output,
                                                                state_seq=state_seq)

            if sample_n < self.N_PLOTTED_ELEMENTS:
                pred_len = outputs.end_ind[b].data.cpu().numpy() + 1 if 'end_ind' in outputs else input_len
                pred_len_seq, _ = self._dense_rec_module.get_sample_with_len(b, pred_len, outputs, inputs,
                                                                             self._pruning_scheme)
                self.evaluation_buffer.rand_seqs[b].append(pred_len_seq.data.cpu().numpy())

    def compute_metrics(self, b, gen_seq, input_seq, outputs, sample_n):
        input_seq = input_seq.detach().cpu().numpy()
        gen_seq = gen_seq.detach().cpu().numpy()
        full_gen_seq = torch.stack([n.subgoal.images[b] for n in outputs.tree.depth_first_iter()]) \
            .detach().cpu().numpy() if 'tree' in outputs \
                                       and outputs.tree.subgoals is not None else gen_seq
        self.evaluation_buffer.mse[b, sample_n] = mse(gen_seq, input_seq)
        if 'psnr' in self.evaluation_buffer:
            self.evaluation_buffer.psnr[b, sample_n] = psnr(gen_seq, input_seq)
        if 'ssim' in self.evaluation_buffer:
            self.evaluation_buffer.ssim[b, sample_n] = ssim(gen_seq, input_seq)
        return full_gen_seq, gen_seq

    @timed("Eval time for batch: ")
    def eval(self, inputs, outputs, model):
        self._erase_eval_buffer()
        if self._top_of_100:
            for n in range(self._top_of):
                outputs = model(inputs)
                self.eval_single(inputs, outputs, sample_n=n)
        else:
            self.eval_single(inputs, outputs)
        self._flush_eval_buffer()

    def _flush_eval_buffer(self):
        if self.full_evaluation is None:
            self.full_evaluation = self.evaluation_buffer
        else:
            dict_concat(self.full_evaluation, self.evaluation_buffer)

    def dump_results(self, it):
        self.dump_metrics(it)
        if self.use_images:
            self.dump_seqs(it)
            if 'matching_outputs' in self.full_evaluation.gen_images[0] \
                    and self.full_evaluation.gen_images[0].matching_outputs is not None:
                self.dump_matching_vis(it)
        self.reset()

    def dump_trees(self, it):
        no_pruning = lambda x, b: False     # show full tree, not pruning anything
        img_dict = self.full_evaluation.gen_images[0]
        plot_matched = img_dict.outputs.tree.match_eval_idx is not None
        assert Evaluator.N_PLOTTED_ELEMENTS <= len(img_dict.gen_images)   # can currently only max plot as many trees as in batch

        def make_padded_seq_img(tensor, target_width, prepend=0):
            assert len(tensor.shape) == 4     # assume [n_frames, channels, res, res]
            n_frames, channels, res, _ = tensor.shape
            seq_im = np.transpose(tensor, (1, 2, 0, 3)).reshape(channels, res, n_frames * res)
            concats = [np.zeros((channels, res, prepend * res), dtype=np.float32)] if prepend > 0 else []
            concats.extend([seq_im, np.zeros((channels, res, target_width - seq_im.shape[2] - prepend * res), dtype=np.float32)])
            seq_im = np.concatenate(concats, axis=-1)
            return seq_im

        with self._logger.log_to('trees', it, 'image'):
            tree_imgs = plot_pruned_tree(img_dict.outputs.tree, no_pruning, plot_matched).detach().cpu().numpy()
            for i in range(Evaluator.N_PLOTTED_ELEMENTS):
                im = tree_imgs[i]
                if plot_matched:
                    gt_seq_im = make_padded_seq_img(img_dict.gt_seq[i], im.shape[-1])
                    pred_seq_im = make_padded_seq_img(img_dict.gen_images[i], im.shape[-1], prepend=1)  # prepend for cropped first frame
                    im = np.concatenate((gt_seq_im, im, pred_seq_im), axis=1)
                im = np.transpose(im, [1, 2, 0])
                self._logger.log(im)

    def dump_metrics(self, it):
        with self._logger.log_to('results', it, 'metric'):
            best_idxs = self._get_best_idxs(self.full_evaluation[self._top_comp_metric])
            print_st = []
            for metric in sorted(self.full_evaluation):
                vals = self.full_evaluation[metric]
                if metric in ['psnr', 'ssim', 'mse']:
                    if metric not in self.evaluation_buffer: continue
                    best_vals = batchwise_index(vals, best_idxs)
                    print_st.extend([best_vals.mean(), best_vals.std(), vals.std(axis=1).mean()])
                    self._logger.log(metric, vals if self._top_of_100 else None, best_vals)
            print(*print_st, sep=',')

    def dump_seqs(self, it):
        """Dumps all predicted sequences and all ground truth sequences in separate .npy files"""
        DUMP_KEYS = ['gt_seq', 'gen_images', 'full_gen_seq']
        batch = len(self.full_evaluation.gen_images)
        _, c, h, w = self.full_evaluation.gen_images[0].gt_seq.shape
        stacked_seqs = AttrDict()

        for key in DUMP_KEYS:
            if key == 'full_gen_seq':
                time = max([i[key].shape[0] for i in self.full_evaluation.gen_images])
            else:
                time = self.full_evaluation.gen_images[0]['gt_seq'].shape[0] - 1
            stacked_seqs[key] = np.zeros((batch, time, c, h, w), dtype=self.full_evaluation.gen_images[0][key].dtype)
        for b, seqs in enumerate(self.full_evaluation.gen_images):
            stacked_seqs['gt_seq'][b] = seqs['gt_seq'][1:]  # skip the first (conditioning frame)
            stacked_seqs['gen_images'][b, :seqs['gen_images'].shape[0]] = seqs['gen_images']
            stacked_seqs['full_gen_seq'][b, :seqs['full_gen_seq'].shape[0]] = seqs['full_gen_seq']
        for b, seqs in enumerate(self.full_evaluation.rand_seqs[:self.N_PLOTTED_ELEMENTS]):
            key = 'seq_samples_{}'.format(b)
            time = self.full_evaluation.gen_images[0]['gt_seq'].shape[0] - 1
            stacked_seqs[key] = np.zeros((self.N_PLOTTED_ELEMENTS, time, c, h, w), dtype=self.full_evaluation.rand_seqs[0][0].dtype)
            for i, seq_i in enumerate(seqs):
                stacked_seqs[key][i, :seq_i.shape[0]] = seq_i[:time]
        for key in DUMP_KEYS:
            with self._logger.log_to(key, it, 'array'):
                self._logger.log(stacked_seqs[key])
        self.dump_gifs(stacked_seqs, it)

        if self._hp.n_rooms is not None and self.full_evaluation.gen_images[0].state_seq is not None:
            self.dump_traj_overview(it)

    def dump_matching_vis(self, it):
        """Dumps some visualization of the matching procedure."""
        with self._logger.log_to('matchings', it, 'image'):
            try:
                for i in range(min(Evaluator.N_PLOTTED_ELEMENTS, self.full_evaluation.gen_images.shape[0])):
                    im = self._dense_rec_module.eval_binding.vis_matching(self.full_evaluation.gen_images[i].matching_outputs)
                    self._logger.log(im)
            except AttributeError:
                print("Binding does not provide matching visualization")
                pass

    def dump_gifs(self, seqs, it):
        """Dumps gif visualizations of pruned and full sequences."""
        with self._logger.log_to('pruned_seq', it, 'gif'):
            im = make_gif([torch.Tensor(seqs.gt_seq), (torch.Tensor(seqs.gen_images))])
            self._logger.log(im)
        with self._logger.log_to('full_gen_seq', it, 'gif'):
            im = make_gif([torch.Tensor(seqs.full_gen_seq)])
            self._logger.log(im)
        for key in seqs:
            if 'seq_samples' in key:
                with self._logger.log_to(key, it, 'gif'):
                    im = make_gif([torch.Tensor(seqs[key])])
                    self._logger.log(im)

    def dump_traj_overview(self, it):
        """Dumps top-down overview of trajectories in Multiroom datasets."""
        from gcp.planning.infra import Multiroom3dEnv
        if self.dummy_env is None:
            self.dummy_env = Multiroom3dEnv({'n_rooms': self._hp.n_rooms}, no_env=True)
        with self._logger.log_to('trajectories', it, 'image'):
            for b in range(min(Evaluator.N_PLOTTED_ELEMENTS, self.full_evaluation.gen_images.shape[0])):
                im = self.dummy_env.render_top_down(self.full_evaluation.gen_images[b].state_seq.data.cpu().numpy())
                self._logger.log(im * 2 - 1)

    def _is_better(self, val, other):
        """Comparison function for different metrics.
           returns True if val is "better" than any of the values in the array other
        """
        if self._top_comp_metric in self.LOWER_IS_BETTER_METRICS:
            return np.all(val <= other)
        elif self._top_comp_metric in self.HIGHER_IS_BETTER_METRICS:
            return np.all(val >= other)
        else:
            raise ValueError("Currently only support comparison on the following metrics: {}. Got {}."
                             .format(self.LOWER_IS_BETTER_METRICS + self.HIGHER_IS_BETTER_METRICS, self._top_comp_metric))

    def _get_best_idxs(self, vals):
        assert len(vals.shape) == 2     # assumes batch in first dimension, N samples in second dim
        if self._top_comp_metric in self.LOWER_IS_BETTER_METRICS:
            return np.argmin(vals, axis=1)
        else:
            return np.argmax(vals, axis=1)


class EvalLogger:
    def __init__(self, log_dir):
        self._log_dir = log_dir
        self.log_target = None
        self.log_type = None
        self.log_tag = None
        self.log_counter = None

    @contextmanager
    def log_to(self, tag, it, type):
        """Sets logging context (e.g. what file to log to)."""
        raise NotImplementedError

    def log(self, *vals):
        """Implements logging within the 'log_to' context."""
        assert self.log_target is not None      # cannot log without 'log_to' context
        if self.log_type == 'metric':
            self._log_metric(*vals)
        elif self.log_type == 'image':
            self._log_img(*vals)
        elif self.log_type == 'array':
            self._log_array(*vals)
        elif self.log_type == 'gif':
            self._log_gif(*vals)
        self.log_counter += 1

    def _log_metric(self, name, vals, best_vals):
        raise NotImplementedError

    def _log_img(self, img):
        raise NotImplementedError

    def _log_array(self, array):
        np.save(os.path.join(self.log_target, "{}_{}.npy".format(self.log_tag, self.log_counter)), array)

    def _log_gif(self, gif):
        pass

    def _make_dump_dir(self, tag, it):
        dump_dir = os.path.join(self._log_dir, '{}/it_{}'.format(tag, it))
        if not os.path.exists(dump_dir): os.makedirs(dump_dir)
        return dump_dir


class FileEvalLogger(EvalLogger):
    """Logs evaluation results on disk."""
    @contextmanager
    def log_to(self, tag, it, type):
        """Creates logging file."""
        self.log_type, self.log_tag, self.log_counter = type, tag, 0
        if type == 'metric':
            self.log_target = open(os.path.join(self._log_dir, '{}_{}.txt'.format(tag, it)), 'w')
        elif type == 'image' or type == 'array':
            self.log_target = self._make_dump_dir(tag, it)
        elif type == 'gif':
            self.log_target = 'no log'
        else:
            raise ValueError("Type {} is not supported for logging in eval!".format(type))
        yield
        if type == 'metric':
            self.log_target.close()
        self.log_target, self.log_type, self.log_tag, self.log_counter = None, None, None, None

    def _log_metric(self, name, vals, best_vals):
        str = 'mean {} {}, standard error of the mean (SEM) {}'.format(name, best_vals.mean(), best_vals.std())
        str += ', mean std of 100 samples {}\n'.format(vals.std(axis=1).mean()) if vals is not None else '\n'
        self.log_target.write(str)
        print(str)

    def _log_img(self, img):
        #assert -1.0 <= img.min() and img.max() <= 1.0   # expect image to be in range [-1...1]
        imsave(os.path.join(self.log_target, "{}_{}.png".format(self.log_tag, self.log_counter)), (img + 1) / 2)


class TBEvalLogger(EvalLogger):
    """Logs evaluation results to Tensorboard."""
    def __init__(self, log_dir, tb_logger):
        super().__init__(log_dir)
        self._tb_logger = tb_logger
        self.log_step = None

    @contextmanager
    def log_to(self, tag, it, type):
        self.log_type, self.log_tag, self.log_counter, self.log_step = type, tag, 0, it
        if type == 'array':
            self.log_target = self._make_dump_dir(tag, it)
        else:
            self.log_target = 'TB'
        yield
        self.log_target, self.log_type, self.log_tag, self.log_counter, self.log_step = None, None, None, None, None

    def _log_metric(self, name, vals, best_vals):
        self._tb_logger.log_scalar(best_vals.mean(), self.group_tag + '/metric/{}/top100_mean'.format(name), self.log_step, '')
        self._tb_logger.log_scalar(best_vals.std(), self.group_tag + '/verbose/{}/top100_std'.format(name), self.log_step, '')
        if vals is not None:
            self._tb_logger.log_scalar(vals.mean(), self.group_tag + '/verbose/{}/all100_mean'.format(name), self.log_step, '')
            self._tb_logger.log_scalar(vals.std(axis=1).mean(), self.group_tag + '/verbose/{}/all100_std'.format(name), self.log_step, '')

    def _log_img(self, img):
        #assert -1.0 <= img.min() and img.max() <= 1.0  # expect image to be in range [-1...1]
        if not isinstance(img, torch.Tensor): img = torch.tensor(img)
        img = (img.permute(2, 0, 1) + 1) / 2
        self._tb_logger.log_images(img[None], self.group_tag + '/{}'.format(self.log_counter), self.log_step, '')

    def _log_gif(self, gif):
        self._tb_logger.log_video(gif, self.group_tag + '/{}'.format(self.log_counter), self.log_step, '')

    @property
    def group_tag(self):
        assert self.log_tag is not None     # need to set logging context first
        return 'eval/{}'.format(self.log_tag)
