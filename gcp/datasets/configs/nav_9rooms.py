from blox import AttrDict
from blox.torch.ops import ten2ar
import numpy as np
import numbers
from gcp.datasets.data_loader import MazeTopRenderedGlobalSplitVarLenVideoDataset


class Nav9Rooms(MazeTopRenderedGlobalSplitVarLenVideoDataset):
    n_rooms = 9
    
    @classmethod
    def render_maze_trajectories(cls, states, end_inds, color, n_logged_samples=3, bckgrds=None):
        from gcp.planning.infra.envs.miniworld_env.multiroom3d.multiroom3d_env import Multiroom3dEnv
        dummy_env = Multiroom3dEnv({'n_rooms': cls.n_rooms}, no_env=True)
        if bckgrds is None:
            bckgrds = [None] * n_logged_samples
            
        if isinstance(color[0], numbers.Number):
            color = [color] * n_logged_samples

        imgs = []
        for i, end_ind in zip(range(n_logged_samples), end_inds):
            state_seq = ten2ar(states[i][:end_ind + 1])
            # if state_seq.shape[0] < 2:
            #     if bckgrds[i] is not None:
            #         imgs.append(bckgrds[i])
            #     continue
            imgs.append(dummy_env.render_top_down(state_seq, color=color[i], background=bckgrds[i]))
        return np.stack(imgs)

    @classmethod
    def render_trajectory(cls, outputs, inputs, predictions, end_inds, n_logged_samples=3):
        # render ground truth trajectories
        im = cls.render_maze_trajectories(inputs.traj_seq_states, inputs.end_ind, (0, 1.0, 0),  # green
                                                n_logged_samples=n_logged_samples)
    
        # render predicted trajectory on top
        color = np.asarray((1.0, 0, 0))  # red
    
        if 'tree' in outputs and 'match_dist' in outputs.tree.subgoals:
            # Color bottleneck frames
            bottleneck_frames = ten2ar(outputs.tree.bf.match_dist.argmax(2)[:, :7])
            end_inds_np = ten2ar(end_inds)
            colors = []
            for i in range(n_logged_samples):
                bottleneck_frames[i, bottleneck_frames[i] > end_inds_np[i]] = end_inds_np[i]
                color_seq = color[None].repeat(end_inds_np[i] + 1, 0)
                color_seq[bottleneck_frames[i]] = color_seq[bottleneck_frames[i]] * 0.5
                color_seq[bottleneck_frames[i] - 1] = color_seq[bottleneck_frames[i] - 1] * 0.5
                colors.append(color_seq)
            color = colors
    
        im = cls.render_maze_trajectories(predictions, end_inds, color, bckgrds=im)
        return {'image': im}


config = AttrDict(
    dataset_spec=AttrDict(
        max_seq_len=100,
        dataset_class=Nav9Rooms,
        split=AttrDict(train=0.994, val=0.006, test=0.00),
    ),
    n_rooms=9,
    crop_window=40,
)
