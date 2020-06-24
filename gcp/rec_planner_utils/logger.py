import os

import copy
import matplotlib.pyplot as plt
import numpy as np
import gcp
import torch
import torchvision
from tensorboardX import SummaryWriter
import numbers

from blox.tensor.ops import broadcast_final, find_tensor
from blox.torch.ops import ten2ar

from gcp.rec_planner_utils.utils import get_pad_mask
from gcp.rec_planner_utils.vis_utils import \
    plot_hierarchical_match_dists, plot_gt_matching_overview, plot_val_tree, plot_gt_action_matching_overview, \
    plot_pruned_seqs, plot_balanced_tree, plot_planning_overview, plot_pruned_tree, make_gif, plot_graph, param, \
    plot_demo_following_overview, fig2img
from gcp.rec_planner_utils.vis_utils import plot_inverse_model_actions, plot_policy_actions



class Logger:
    def __init__(self, log_dir, n_logged_samples=10, summary_writer=None, fps=4):
        self._log_dir = log_dir
        self._n_logged_samples = n_logged_samples
        self.fps = fps
        if summary_writer is not None:
            self._summ_writer = summary_writer
        else:
            self._summ_writer = SummaryWriter(log_dir, max_queue=1, flush_secs=1)

    def _loop_batch(self, fn, name, val, *argv, **kwargs):
        """Loops the logging function n times."""
        for log_idx in range(min(self._n_logged_samples, len(val))):
            name_i = os.path.join(name, "_%d" % log_idx)
            fn(name_i, val[log_idx], *argv, **kwargs)
     
    @staticmethod
    def _check_size(val, size):
        if isinstance(val, torch.Tensor) or isinstance(val, np.ndarray):
            assert len(val.shape) == size, "Size of tensor does not fit required size, {} vs {}".format(len(val.shape), size)
        elif isinstance(val, list):
            assert len(val[0].shape) == size-1, "Size of list element does not fit required size, {} vs {}".format(len(val[0].shape), size-1)
        else:
            raise NotImplementedError("Input type {} not supported for dimensionality check!".format(type(val)))
        # if (val[0].numel() > 1e9):
        #     print("Logging very large image with size {}px.".format(max(val[0].shape[1], val[0].shape[2])))
            #raise ValueError("This might be a bit too much")

    def log_scalar(self, scalar, name, step, phase):
        self._summ_writer.add_scalar('{}_{}'.format(name, phase), scalar, step)

    def log_scalars(self, scalar_dict, group_name, step, phase):
        """Will log all scalars in the same plot."""
        self._summ_writer.add_scalars('{}_{}'.format(group_name, phase), scalar_dict, step)

    def log_images(self, image, name, step, phase):
        self._check_size(image, 4)   # [N, C, H, W]
        if image[0].numel() > 3e9:
            print('skipping logging a giant image with {}px'.format(image[0].numel()))
            return
        
        self._loop_batch(
            self._summ_writer.add_image, '{}_{}'.format(name, phase), image, step)

    def log_video(self, video_frames, name, step, phase):
        assert len(video_frames.shape) == 4, "Need [T, C, H, W] input tensor for single video logging!"
        if not isinstance(video_frames, torch.Tensor): video_frames = torch.tensor(video_frames)
        #video_frames = torch.transpose(video_frames, 0, 1)  # tbX requires [C, T, H, W] <- not in tbX >= 1.6
        video_frames = video_frames.unsqueeze(0)    # add an extra dimension to get grid of size 1
        self._summ_writer.add_video('{}_{}'.format(name, phase), video_frames, step, fps=self.fps)

    def log_videos(self, video_frames, name, step, phase):
        assert len(video_frames.shape) == 5, "Need [N, T, C, H, W] input tensor for video logging!"
        video_frames = video_frames.unsqueeze(1)    # add an extra dimension after batch to get grid of size 1
        self._loop_batch(self._summ_writer.add_video, '{}_{}'.format(name, phase), video_frames, step, fps=self.fps)

    def log_image_grid(self, images, name, step, phase, nrow=8):
        assert len(images.shape) == 4, "Image grid logging requires input shape [batch, C, H, W]!"
        img_grid = torchvision.utils.make_grid(images, nrow=nrow)
        if img_grid.min() < 0:
            print("warning, image not rescaled!")
            img_grid = (img_grid + 1) / 2
        
        self._summ_writer.add_image('{}_{}'.format(name, phase), img_grid, step)

    def log_video_grid(self, video_frames, name, step, phase, fps=3):
        assert len(video_frames.shape) == 5, "Need [N, T, C, H, W] input tensor for video logging!"
        self._summ_writer.add_video('{}_{}'.format(name, phase), video_frames, step, fps=fps)

    def log_figures(self, figure, name, step, phase):
        """figure: matplotlib.pyplot figure handle"""
        assert figure.shape[0] > 0, "Figure logging requires input shape [batch x figures]!"
        self._loop_batch(self._summ_writer.add_figure, '{}_{}'.format(name, phase), figure, step)

    def log_figure(self, figure, name, step, phase):
        """figure: matplotlib.pyplot figure handle"""
        self._summ_writer.add_figure('{}_{}'.format(name, phase), figure, step)
        
    def log_graph(self, array, name, step, phase):
        """figure: matplotlib.pyplot figure handle"""
        im = plot_graph(array)
        self._summ_writer.add_image('{}_{}'.format(name, phase), im.transpose(2, 0, 1), step)

    def dump_scalars(self, log_path=None):
        log_path = os.path.join(self._log_dir, "scalar_data.json") if log_path is None else log_path
        self._summ_writer.export_scalars_to_json(log_path)


class HierarchyLogger(Logger):
    def __init__(self, log_dir, hp, max_seq_len, n_logged_samples=3, summary_writer=None, fps=4):
        Logger.__init__(self, log_dir, n_logged_samples, summary_writer, fps)
        self._gamma_width = 4       # width of the gamma visualization in px
        self._hp = hp
        self.max_seq_len = max_seq_len
        self.dummy_env = None

    def render(self, tensor):
        return tensor

    def log_hierarchy_image(self, model_output, inputs, name, step, phase):
        """Builds an image depicting the predicted hierarchy."""
        reference = find_tensor(inputs)
        batch_size, channels, im_height, _ = inputs.I_0_image.shape
        N = self.max_seq_len
        assert batch_size >= self._n_logged_samples
        depth = model_output.tree.depth
        
        if 'gamma' in model_output.tree.subgoals.keys():
            gamma_width = self._gamma_width
        else:
            gamma_width = 0
        
        level_height = im_height + gamma_width
        image = 0.7 * torch.ones((self._n_logged_samples, channels, depth * level_height, N * im_height),
                                 dtype=torch.float32, device=reference.device)
        for level in range(depth):
            # initialize gamma "strips" to 0
            image[:, :, level * level_height : level * level_height + gamma_width] = 0.0
    
        def tile_gamma(gamma):
            return gamma[:, None].repeat(1, im_height).view(-1)

        # TODO vectorize the batch computation
        for segment in model_output.tree.depth_first_iter():
            subgoal = segment.subgoal
            if not subgoal.keys(): break
            level = depth - segment.depth
            # visualize gamma on this depth level
            for batch_idx in range(self._n_logged_samples):
                if 'done_mask' in segment and (segment.done_mask is None or not segment.done_mask[batch_idx]):
                    time_idx = int(subgoal.ind[batch_idx])
                    if gamma_width != 0:
                        image[batch_idx, :, level * level_height:level * level_height + gamma_width] += \
                            tile_gamma(subgoal.gamma[batch_idx])

                    if 'images' in subgoal.keys():
                        image[batch_idx,
                              :,
                              level * level_height + gamma_width:(level+1) * level_height,
                              time_idx * im_height:(time_idx + 1) * im_height] = self.render(subgoal.images[batch_idx])
                
        def flatten_seq(seq):
            return seq[:self._n_logged_samples].transpose(1, 2).transpose(2, 3).\
                reshape(self._n_logged_samples, channels, im_height, N * im_height)
            
        def mask_extra(seq):
            return seq * broadcast_final(get_pad_mask(model_output.end_ind, N), seq)
        
        if 'demo_seq' in inputs.keys():
            input_image = flatten_seq(inputs.demo_seq_images)
        else:
            input_image = torch.zeros_like(image[:,:,:level_height])
            input_image[:,:,:,:im_height] = inputs.I_0_image[:self._n_logged_samples]
            input_image[:,:,:,-im_height:] = inputs.I_g_image[:self._n_logged_samples]
        image = torch.cat([input_image, image], dim=2)
            
        if 'dense_rec' in model_output.keys() and 'images' in model_output.dense_rec.keys()\
                and not 'p_n_hat' in model_output.tree.subgoals:     # don't log the dense rec here if it is pruned
            dense_images = mask_extra(self.render(model_output.dense_rec.images))
            image = torch.cat([image, flatten_seq(dense_images)], dim=2)

        if 'soft_matched_estimates' in model_output.keys():
            soft_estimates = mask_extra(self.render(model_output.soft_matched_estimates))
            image = torch.cat([image, flatten_seq(soft_estimates)], dim=2)

        image = (image + 1)/2   # rescale back to 0-1 range
        self.log_images(image, name, step, phase)
        return image

    def _log_plot_img(self, img, name, step, phase):
        if not isinstance(img, torch.Tensor): img = torch.tensor(img)
        img = img.float().permute(0, 3, 1, 2)
        self.log_images(img, name, step, phase)

    def log_gif(self, imgs, name, step, phase):
        if isinstance(imgs, list): imgs = np.concatenate(imgs)
        if not isinstance(imgs, torch.Tensor): imgs = torch.tensor(imgs)
        imgs = imgs.float().permute(0, 3, 1, 2)
        self.log_video(imgs, name + "_gif", step, phase)

    def log_match_dists(self, model_output, name, step, phase):
        self._log_plot_img(plot_hierarchical_match_dists(model_output), name, step, phase)

    def log_gt_match_overview(self, model_output, inputs, name, step, phase):
        self._log_plot_img(plot_gt_matching_overview(model_output, inputs), name, step, phase)

    def log_gt_action_match_overview(self, model_output, inputs, hp, name, step, phase):
        self._log_plot_img(plot_gt_action_matching_overview(model_output, inputs, hp), name, step, phase)

    def log_attention_overview(self, model_output, inputs, name, step, phase):
        self._log_plot_img(plot_gt_matching_overview(model_output, inputs, plot_attr='gamma'), name, step, phase)

    def log_val_tree(self, model_output, inputs, name, step, phase):
        self._log_plot_img(plot_val_tree(model_output, inputs), name, step, phase)

    def log_balanced_tree(self, model_output, element, name, step, phase):
        with param(n_logged_samples=self._n_logged_samples):
            im = plot_balanced_tree(model_output, element).transpose((0, 2, 3, 1))
        self._log_plot_img(im, name, step, phase)  # show from first, last and predicted

    def log_pruned_tree(self, model_output, name, step, phase):
        self._log_plot_img(plot_pruned_tree(model_output.tree).permute(0, 2, 3, 1), name, step, phase)

    def log_pruned_pred(self, model_output, inputs, name, step, phase):
        im, seq = plot_pruned_seqs(model_output, inputs)
        self._log_plot_img(im, name, step, phase)
        self.log_gif(seq, name, step, phase)

    def log_planning_overview(self, planner_outputs=None, exec_seq=None, goal=None, _name=None, _step=None, phase=None):
        im = plot_planning_overview(planner_outputs, exec_seq, goal)
        self._log_plot_img(im, _name, _step, phase)

    def log_demo_following_overview(self, demo_seq=None, exec_seq=None, goal=None, _name=None, _step=None, phase=None):
        im = plot_demo_following_overview(demo_seq, exec_seq, goal)
        self._log_plot_img(im, _name, _step, phase)

    def log_dense_gif(self, model_output, inputs, name, step, phase):
        """Logs the dense reconstruction """
    
        rows = [model_output.dense_rec.images]
        if phase == 'train':
            rows = [inputs.demo_seq] + rows
            
        self.log_rows_gif(rows, name, step, phase)
        # self.log_loss_gif(model_output.dense_rec.images, inputs.demo_seq, name, step, phase)

    def log_rows_gif(self, rows, name, step, phase):
        """ Logs a gif with several rows
        
        :param rows: a list of tensors batch x time x channel x height x width
        :param name:
        :param step:
        :param phase:
        :return:
        """
        im = make_gif(rows)
        self.log_video(im, name, step, phase)

    def log_loss_gif(self, estimates, targets, name, step, phase):
        """Logs gifs showing a target and a ground truth sequence side by side,
         shows ground truth sequence at training time too."""
        
        estimate_imgs = self.render((estimates + 1) / 2)
        target_imgs = self.render((targets + 1) / 2)
        if phase == "train":
            # concat target images
            seq_shape = target_imgs.shape
            padding = torch.zeros(seq_shape[:3] + (2,) + seq_shape[4:], dtype=target_imgs.dtype,
                                  device=target_imgs.device)
            plot_imgs = torch.cat([target_imgs, padding, estimate_imgs], dim=3)
        else:
            plot_imgs = estimate_imgs
        plot_imgs = plot_imgs[:5]
        batch, time, channels, height, width = plot_imgs.shape
        plot_imgs = plot_imgs.permute(1, 2, 3, 0, 4).reshape(time, channels, height, width * batch)
        self.log_video(plot_imgs, name, step, phase)

    def log_topdown_traj_overview(self, inputs, name, step, phase):
        from gcp.infra.envs.miniworld_env.multiroom3d.multiroom3d_env import Multiroom3dEnv
        if self.dummy_env is None:
            self.dummy_env = Multiroom3dEnv({'n_rooms': self._hp.n_rooms}, no_env=True)
        im = np.stack([self.dummy_env.render_top_down(inputs.demo_seq_states[i, :end_ind].data.cpu().numpy()) for i, end_ind in
                                                      zip(range(self._n_logged_samples), inputs.end_ind)])
        self._log_plot_img(im, name, step, phase)

    def log_single_topdown_traj(self, traj, name, step, phase):
        from gcp.infra.envs.miniworld_env.multiroom3d.multiroom3d_env import Multiroom3dEnv
        if self.dummy_env is None:
            self.dummy_env = Multiroom3dEnv({'n_rooms': self._hp.n_rooms}, no_env=True)
        im = self.dummy_env.render_top_down(traj)
        self._log_plot_img(im[None], name, step, phase)

    def log_multiple_topdown_trajs(self, trajs, name, step, phase):
        from gcp.infra.envs.miniworld_env.multiroom3d.multiroom3d_env import Multiroom3dEnv
        if self.dummy_env is None:
            self.dummy_env = Multiroom3dEnv({'n_rooms': self._hp.n_rooms}, no_env=True)
        im = self.dummy_env.render_top_down(trajs[0])
        for traj in trajs[1:]:
            im = self.dummy_env.render_top_down(traj, background=im)
        self._log_plot_img(im[None], name, step, phase)

    def log_maze_topdown(self, model_output, inputs, name, step, phase, predictions=None, end_inds=None):
        """Logs ground truth (green) and predicted (red) trajectory in one frame."""

        # render ground truth trajectories
        im = self.render_maze_trajectories(inputs.demo_seq_states, inputs.end_ind, (0, 1.0, 0))  # green

        # render predicted trajectory on top
        if predictions is None:
            predictions, end_inds = self.get_predictions_from_output(inputs, model_output)
        color = np.asarray((1.0, 0, 0))  # red
        
        if 'tree' in model_output and 'match_dist' in model_output.tree.subgoals:
            # Color bottleneck frames
            bottleneck_frames = ten2ar(model_output.tree.bf.match_dist.argmax(2)[:, :7])
            end_inds_np = ten2ar(end_inds)
            colors = []
            for i in range(self._n_logged_samples):
                color_seq = color[None].repeat(end_inds_np[i] + 1, 0)
                color_seq[bottleneck_frames[i]] = color_seq[bottleneck_frames[i]] * 0.5
                color_seq[bottleneck_frames[i] - 1] = color_seq[bottleneck_frames[i] - 1] * 0.5
                colors.append(color_seq)
            color = colors

        im = self.render_maze_trajectories(predictions, end_inds, color, bckgrds=im)
        self._log_plot_img(im, name, step, phase)

    def render_maze_trajectories(self, states, end_inds, color, bckgrds=None):
        from gcp.infra.envs.miniworld_env.multiroom3d.multiroom3d_env import Multiroom3dEnv
        if self.dummy_env is None:
            self.dummy_env = Multiroom3dEnv({'n_rooms': self._hp.n_rooms}, no_env=True)
        if bckgrds is None:
            bckgrds = [None] * self._n_logged_samples
        if isinstance(color[0], numbers.Number):
            color = [color] * self._n_logged_samples

        imgs = []
        for i, end_ind in zip(range(self._n_logged_samples), end_inds):
            state_seq = ten2ar(states[i][:end_ind + 1])
            # if state_seq.shape[0] < 2:
            #     if bckgrds[i] is not None:
            #         imgs.append(bckgrds[i])
            #     continue
            imgs.append(self.dummy_env.render_top_down(state_seq, color=color[i], background=bckgrds[i]))
        return np.stack(imgs)

        #return np.stack([self.dummy_env.render_top_down(ten2ar(states[i][:end_ind]), color=color[i], background=bckgrds[i])
        #                 for i, end_ind in zip(range(self._n_logged_samples), end_inds) if end_ind >= 1])

    @staticmethod
    def get_predictions_from_output(inputs, model_output):
        if 'pruned_prediction' in model_output:
            predictions = model_output.pruned_prediction
            end_inds = model_output.end_ind
        elif 'images' in model_output.dense_rec:
            predictions = model_output.dense_rec.images
            end_inds = inputs.end_ind
        elif 'soft_matched_estimates' in model_output:
            predictions = model_output.soft_matched_estimates
            end_inds = inputs.end_ind
        else:
            predictions = model_output.tree.df.images
            end_inds = np.ones((predictions.shape[0],), dtype=int) * predictions.shape[1]   # use full length
        return predictions, end_inds

    def log_n_trajs_topdown(self, trajs, name, step, phase, return_img=False):
        """Logs all trajectories in trajs into one image."""
        from gcp.infra.envs.miniworld_env.multiroom3d.multiroom3d_env import Multiroom3dEnv
        if self.dummy_env is None:
            self.dummy_env = Multiroom3dEnv({'n_rooms': self._hp.n_rooms}, no_env=True)
        im = None
        for traj in trajs:
            im = self.dummy_env.render_top_down(traj, background=im)
        if return_img:
            return im
        else:
            self._log_plot_img(im[None], name, step, phase)

    def log_n_trajs_topdown_multiiter_multistep(self, trajs_multiiter_multistep, n_rollouts, name, step, phase):
        im = np.concatenate(
                [np.concatenate([self.log_n_trajs_topdown(trajs.elite_rollouts[:n_rollouts],
                                                          name, step, phase, return_img=True)
                                 for trajs in trajs_multiiter], axis=1)
             for trajs_multiiter in trajs_multiiter_multistep], axis=0)
        self._log_plot_img(im[None], name, step, phase)

    def log_n_trajs_cartgripper(self, trajs, name, step, phase, return_img=False):
        """Logs all trajectories in trajs into one gif."""
        TRAJ_LEN = 100      # all trajectories will get padded to this length
        assert len(trajs) == 1  # can only log a single trajectory at a time for cartgripper!
        from gcp.rec_planner_utils.log_cartgripper_utils import render_cartgripper
        #if trajs[0].shape[-1] == 10:
        #    traj = np.concatenate([trajs[0][:, :1], trajs[0][:, 2:5], trajs[0][:, 6:8], trajs[0][:, -1:]], axis=-1)
        #else:
        #    traj = trajs[0]
        imgs = render_cartgripper(trajs[0])
        imgs = np.transpose(imgs.astype(np.float32) / 255., [0, 3, 1, 2])

        # blend in goal image
        imgs = 0.7 * imgs + 0.3 * np.repeat(imgs[-1:], imgs.shape[0], axis=0)

        # pad to equal length
        if imgs.shape[0] < TRAJ_LEN:
            t, c, w, h = imgs.shape
            imgs = np.concatenate([imgs, np.zeros((TRAJ_LEN-t, c, w, h))])

        if return_img:
            return imgs
        else:
            self.log_video(imgs, name, step, phase)

    def log_n_trajs_cartgripper_multiiter_multistep(self, trajs_multiiter_multistep, n_rollouts, name, step, phase):
        
        def log_0(trajs):
            return self.log_n_trajs_cartgripper(trajs.elite_rollouts[:n_rollouts], name, step, phase, return_img=True)
        
        def log_1(trajs_multiiter):
            return np.concatenate([log_0(trajs) for trajs in trajs_multiiter], axis=-1)
        
        im = np.concatenate([log_1(trajs_multiiter) for trajs_multiiter in trajs_multiiter_multistep], axis=-2)
        
        self.log_video(im, name, step, phase)

    def log_states_2d(self, model_output, inputs, name, step, phase):
        """Logs 2D plot of first 2 state dimensions."""
        predictions, end_inds = self.get_predictions_from_output(inputs, model_output)

        imgs = []
        for i, end_ind_inp, end_ind in zip(range(self._n_logged_samples), inputs.end_ind, end_inds):
            fig = plt.figure()
            gt = inputs.demo_seq_states[i, :end_ind_inp].data.cpu().numpy()
            plt.plot(gt[:, 0], gt[:, 1], 'g')
            pred = predictions[i][:end_ind].data.cpu().numpy()
            plt.plot(pred[:, 0], pred[:, 1], 'r')
            imgs.append(fig2img(fig))
        self._log_plot_img(np.stack(imgs), name, step, phase)

    def log_sawyer(self, model_output, inputs, name, step, phase, demo_data_dir):
        """Logs sawyer from states"""

        from gcp.rec_planner_utils.log_sawyer_utils import render_roboturk_env_joint_actions
        imgs = render_roboturk_env_joint_actions(inputs, model_output)
        imgs = np.transpose(imgs.astype(np.float32)/255., [0, 3, 1, 2])
        self.log_video(imgs, name, step, phase)

    def log_cartgripper(self, model_output, inputs, name, step, phase, demo_data_dir):
        """Logs cartgripper from states"""
        from gcp.rec_planner_utils.log_cartgripper_utils import render_cartgripper
        raw_outputs = self.get_predictions_from_output(inputs, model_output)[0]
        n_samples = len(raw_outputs) if isinstance(raw_outputs, list) else raw_outputs.shape[0]
        for i in range(min(self._n_logged_samples, n_samples)):
            imgs = render_cartgripper(raw_outputs[i].data.cpu().numpy())
            imgs = np.transpose(imgs.astype(np.float32)/255., [0, 3, 1, 2])
            self.log_video(imgs, name+"_{}".format(i), step, phase)

    def log_pred_actions(self, model_output, inputs, name, step, phase):
        self._log_plot_img(plot_inverse_model_actions(model_output, inputs), name, step, phase)


class Mujoco_Renderer():
    def __init__(self, mujoco_xml, hp):
        from mujoco_py import load_model_from_path, MjSim
        mujoco_xml = '/'.join(str.split(gcp.__file__, '/')[:-1]) + '/' + mujoco_xml
        self.sim = MjSim(load_model_from_path(mujoco_xml))
        self._hp = hp

    def render(self, qpos):
        sim_state = self.sim.get_state()
        sim_state.qpos[:2] = qpos
        sim_state.qvel[:] = np.zeros_like(self.sim.data.qvel)
        self.sim.set_state(sim_state)
        self.sim.forward()
        
        subgoal_image = self.sim.render(self._hp.mpar.img_sz, self._hp.mpar.img_sz, camera_name='maincam')
        # plt.imshow(subgoal_image)
        # plt.savefig('test.png')
        return subgoal_image


class HierarchyLoggerTest(Logger):
    def __init__(self, log_dir, hp, max_seq_len, n_logged_samples=10, summary_writer=None):
        super().__init__(log_dir, n_logged_samples, summary_writer)
        self._gamma_width = 4  # width of the gamma visualization in px
        self._hp = hp

    def render(self, tensor):
        return tensor

    def log_hierarchy_image(self, model_output, inputs, name, step, phase):
        """Builds an image depicting the predicted hierarchy."""
        e_0, e_g = inputs.I_0_image, inputs.I_g_image
        batch_size, channels, im_height, self.imwidth = e_0.shape
        assert batch_size >= self._n_logged_samples
        total_depth = model_output.tree.depth
        total_length = np.sum([2**i for i in range(total_depth)])
        n = total_length + 2
        
        image = 0.7 * e_0.new_ones((self._n_logged_samples, channels, (total_depth + 1) * im_height, n * im_height))
        
        image[:, :, 0:im_height, 0:im_height] = e_0[:self._n_logged_samples]
        image[:, :, 0:im_height, -im_height:] = e_g[:self._n_logged_samples]
        
        for i, segment in enumerate(model_output.tree):
            subgoal = segment.subgoal
            depth = 1 + total_depth - segment.depth
            image[:, :, depth * im_height:(1 + depth) * im_height, (i + 1) * im_height:(i + 2) * im_height] = \
                self.render(subgoal.images[:self._n_logged_samples])

        image = (image + 1) / 2  # rescale back to 0-1 range
        self.log_images(image, name, step, phase)
        return image


class HierarchyLoggerLowdimStates(HierarchyLogger, Mujoco_Renderer):
    def __init__(self, *args, **kwargs):
        HierarchyLogger.__init__(self, *args, **kwargs)
        Mujoco_Renderer.__init__(self, self._hp.mujoco_xml, self._hp)

    def render(self, tensor):
        device = tensor.device
        tensor = tensor.data.cpu().numpy()
        sh = tensor.shape
        tensor = np.reshape(tensor, [-1, sh[-1]])
        images = np.stack([Mujoco_Renderer.render(self, state) for state in tensor])
        images = np.reshape(images, sh[:-1] + images.shape[1:])
        images = np.swapaxes(images, -1, -3)
        images = (images/255.0) * 2 - 1
        return torch.Tensor(images, device=device)


class HierarchyLoggerLowdimStatesTest(HierarchyLoggerTest, Mujoco_Renderer):
    def __init__(self, *args, **kwargs):
        HierarchyLogger.__init__(self, *args, **kwargs)
        Mujoco_Renderer.__init__(self, self._hp.mujoco_xml, self._hp)

    def render(self, tensor):
        return Mujoco_Renderer.render(self, tensor)


class InverseModelLogger(Logger, Mujoco_Renderer):
    def __init__(self, log_dir, hp, max_seq_len, n_logged_samples=10, summary_writer=None):
        Logger.__init__(self, log_dir, n_logged_samples, summary_writer)
        if hp.mujoco_xml is not None:
            Mujoco_Renderer.__init__(self, hp.mujoco_xml, hp)
        self._gamma_width = 4  # width of the gamma visualization in px
        self._hp = hp
        self.max_seq_len = max_seq_len

    def _log_plot_img(self, img, name, step, phase):
        if not isinstance(img, torch.Tensor): img = torch.tensor(img)
        img = img.float().permute(0, 3, 1, 2)
        self.log_images(img, name, step, phase)

    def log_pred_actions(self, model_output, inputs, name, step, phase):
        self._log_plot_img(plot_inverse_model_actions(model_output, inputs), name, step, phase)

    def log_pred_states(self, model_output, inputs, name, step, phase):
        imrow = []
        for b in range(self._n_logged_samples):
            imrow.append(self.render(model_output.states[b].cpu().numpy()))
        self._log_plot_img(np.concatenate(imrow, 1), name, step, phase)


class BCModelLogger(Logger):
    def __init__(self, log_dir, hp, max_seq_len, n_logged_samples=10, summary_writer=None, *args, **kwargs):
        Logger.__init__(self, log_dir, n_logged_samples, summary_writer, *args, **kwargs)
        self._gamma_width = 4  # width of the gamma visualization in px
        self._hp = hp
        self.max_seq_len = max_seq_len

    def _log_plot_img(self, img, name, step, phase):
        if not isinstance(img, torch.Tensor): img = torch.tensor(img)
        img = img.float().permute(0, 3, 1, 2)
        self.log_images(img, name, step, phase)

    def log_pred_actions(self, model_output, inputs, name, step, phase):
        n_logged_actions = min(model_output.actions.shape[1], 50)
        self._log_plot_img(plot_policy_actions(model_output.actions[:, :n_logged_actions], inputs.actions[:, :n_logged_actions], inputs.demo_seq_images[:, :n_logged_actions+1]), name, step, phase)


if __name__ == "__main__":
    logger = Logger(log_dir="./summaries")
    for step in range(10):
        print("Running step %d" % step)
        dummy_data = torch.rand([32, 10, 3, 64, 64])
        logger.log_scalar(dummy_data[0, 0, 0, 0, 0], name="scalar", step=step, phase="train")
        logger.log_scalars({
            'test1': dummy_data[0, 0, 0, 0, 0],
            'test2': dummy_data[0, 0, 0, 0, 1],
            'test3': dummy_data[0, 0, 0, 0, 2]
        }, group_name="scalar_group", step=step, phase="train")
        logger.log_images(dummy_data[:, 0], name="image", step=step, phase="train")
        logger.log_video(dummy_data, name="video", step=step, phase="train")
        logger.log_video_grid(dummy_data, name="video_grid", step=step, phase="train")
        fig = plt.figure()
        plt.plot(dummy_data.data.numpy()[:, 0, 0, 0, 0])
        logger.log_figures(np.asarray([fig for _ in range(10)]), name="figure", step=step, phase="train")
    logger.dump_scalars()
    print("Done!")


