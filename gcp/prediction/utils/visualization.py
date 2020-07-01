from contextlib import contextmanager

import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import numbers
from torch.nn.utils.rnn import pad_sequence as pad_sequence

from blox.tensor.ops import batchwise_index
from blox.tensor import ndim

PLOT_BINARY_DISTS = True    # if True, plots greyscale tiles instead of line plots for matching functions

class Params():
    """ Singleton holding visualization params """
    n_logged_samples = 3
    
PARAMS = Params()


@contextmanager
def param(**kwargs):
    """ A context manager that sets global params to specified values for the context duration """
    # Update values
    old_kwargs = {}
    for name, value in kwargs.items():
        old_kwargs[name] = getattr(Params, name)
        setattr(Params, name, value)
        
    yield
    
    # Reset old values
    for name, value in old_kwargs.items():
        setattr(Params, name, value)
    

def fig2img(fig):
    """Converts a given figure handle to a 3-channel numpy image array."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis=2)
    w, h, d = buf.shape
    return np.array(Image.frombytes("RGBA", (w, h), buf.tostring()), dtype=np.float32)[:, :, :3] / 255.


def plot_greyscale_dist(dist, h, w):
    n_tiles = dist.shape[0]
    tile_width = int(w / n_tiles)
    tiled_im = np.repeat(np.repeat(np.repeat(dist, tile_width, axis=0)[None], h, axis=0)[..., None], 3, axis=-1)
    return tiled_im


def plot_dists(dists, h=400, w=400, dpi=10, linewidth=1.0):
    if PLOT_BINARY_DISTS and len(dists) == 1:
        return plot_greyscale_dist(dists[0], h, w)
    COLORS = ['red', 'blue', 'green']
    assert len(dists) <= 3      # only have 3 different colors for now, add more if necessary!
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    for dist, color in zip(dists, COLORS[:len(dists)]):
        plt.plot(dist, color=color, linewidth=linewidth)
    plt.ylim(0, 1)
    plt.xlim(0, dist.shape[0]-1)
    plt.tight_layout()
    fig_img = fig2img(fig)
    plt.close(fig)
    return fig_img


def plot_graph(array, h=400, w=400, dpi=10, linewidth=3.0):
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    if isinstance(array, torch.Tensor):
        array = array.cpu().numpy()
    plt.xlim(0, array.shape[0] - 1)
    plt.xticks(fontsize=100)
    plt.yticks(fontsize=100)
    plt.plot(array)
    plt.grid()
    plt.tight_layout()
    fig_img = fig2img(fig)
    plt.close(fig)
    return fig_img


def tensor2np(tensor, n_logged_samples=None):
    if tensor is None: return tensor
    if n_logged_samples is None: return tensor.data.cpu().numpy()
    return tensor[:n_logged_samples].data.cpu().numpy()


def visualize(tensor):
    """ Visualizes the state. Returns blank image as default, otherwise calls the dataset method"""
    n_logged_samples = PARAMS.n_logged_samples
    array = np.ones(tensor.shape[:-1] + (3, PARAMS.hp.img_sz, PARAMS.hp.img_sz), dtype=np.float32)[:n_logged_samples]
    
    if hasattr(PARAMS, 'visualize'):
        array = PARAMS.visualize(tensor, array, PARAMS.hp) # 'visualize' is set in train_planner_mode
        
    return array


def imgtensor2np(tensor, n_logged_samples=None, gt_images=False):
    if tensor is None: return tensor
    
    if not PARAMS.hp.use_convs and not gt_images:
        with param(n_logged_samples=n_logged_samples):
            return visualize(tensor[:n_logged_samples])
        
    return (tensor2np(tensor, n_logged_samples) + 1)/2


def np2imgtensor(array, device, n_logged_samples=None):
    if array is None: return array
    if n_logged_samples is not None: array = array[:n_logged_samples]
    return torch.tensor(array * 2 - 1)


def action2img(action, res, channels):
    action_scale = 50   # how much action arrows should get magnified
    assert action.size == 2   # can only plot 2-dimensional actions
    img = np.zeros((res, res, channels), dtype=np.float32).copy()
    start_pt = res/2 * np.ones((2,))
    end_pt = start_pt + action * action_scale * (res/2 - 1) * np.array([1, -1])     # swaps last dimension
    np2pt = lambda x: tuple(np.asarray(x, int))
    img = cv2.arrowedLine(img, np2pt(start_pt), np2pt(end_pt), (255, 255, 255), 1, cv2.LINE_AA, tipLength=0.2)
    return img * 255.0


def batch_action2img(actions, res, channels):
    batch, seq_len, _ = actions.shape
    im = np.empty((batch, seq_len, res, res, channels), dtype=np.float32)
    for b in range(batch):
        for s in range(seq_len):
            im[b, s] = action2img(actions[b, s], res, channels)
    return im


class channel_last:
    """ A function decorator that transposes the input and output image if necessary
     
     The first input to the function has to be the image, and the function must output an image back
     """
    
    def __init__(self):
        """ Decorator parameters """
        pass
    
    def __call__(self, func):
        """ Wrapping """
        
        def wrapper(img, *args, **kwargs):
            transpose = self.is_channel_first(img.shape[-3:])
            
            if transpose:
                sh_offset = len(img.shape) - 3
                offset_order = lambda x: np.concatenate([np.arange(sh_offset), np.array(x) + sh_offset])
                img = ndim.permute(img, offset_order([1, 2, 0]))
                
            result = func(img, *args, **kwargs)
            
            if transpose:
                result = ndim.permute(result, offset_order([2, 0, 1]))
            return result
        
        return wrapper
    
    @staticmethod
    def is_channel_first(shape):
        # The allowed channel dimesions are [1, 3]
    
        # If the last dim is not an allowed channel dimension
        if shape[2] not in [1, 3]:
            assert shape[0] in [1, 3]
            return True
    
        # If the first dim is not an allowed channel dimension
        if shape[0] not in [1, 3]:
            return False
    
        # If the last dim is image size but the first is not
        if hasattr(PARAMS, 'hp') and shape[2] == PARAMS.hp.img_sz and shape[0] != PARAMS.hp.img_sz:
            return True


@ndim.torched
@channel_last()
def draw_frame(img, prob):
    """
    
    :param img: array dims x width x height x colors
    :param prob: array dims
    :return:
    """
    if img.shape[2] == 1: return img
    if isinstance(prob, numbers.Number):
        prob = np.full_like(img[..., 0, 0, 0], prob)
    img = ndim.copy(img)
    
    cmap = cm.get_cmap('RdYlGn')
    rgb = cmap(prob[..., None, None])[..., :3]
    img[..., :, :2, :], img[..., :, -2:, :] = rgb, rgb
    img[..., :2, :, :], img[..., -2:, :, :] = rgb, rgb
    return img


def framed_action2img(action, prob, res, channel):
    """Draws frame around action image indicating the probability of usage."""
    img = action2img(action, res, channel)
    img = draw_frame(img, prob)
    return img


def sort_actions_depth_first(model_output, attrs, n_logged_samples):
    """Sorts actions in depth-first ordering. attrs is a list with attribute names for the left/right
       distributions that should be sorted."""
    assert len(attrs) == 2  # need one left and one right action attribute
    tree = model_output.tree
    n_sg = (2 ** tree.depth) - 1
    dist_shape = [n_logged_samples, 2 * n_sg + 1]
    dist_shape = dist_shape + list(tree.subgoals[attrs[0]].shape[2:]) if len(tree.subgoals[attrs[0]].shape) > 2 else dist_shape
    match_dists = torch.zeros(dist_shape, device=tree.subgoals[attrs[0]].device)
    for i, segment in enumerate(tree.depth_first_iter()):
        match_dists[:, 2 * (i + 1) - 2 ** (segment.depth - 1) - 1] = segment.subgoal[attrs[0]][:n_logged_samples]
        match_dists[:, 2 * (i + 1) + 2 ** (segment.depth - 1) - 1] = segment.subgoal[attrs[1]][:n_logged_samples]
    return match_dists


def plot_balanced_tree(model_output, elem="images"):
    """Plots all subgoals of the tree in a balanced format."""
    tree = model_output.tree
    max_depth = tree.depth
    batch = tree.subgoals[elem].shape[0]
    res = PARAMS.hp.img_sz
    n_logged_samples = PARAMS.n_logged_samples
    n_logged_samples = batch if n_logged_samples is None else n_logged_samples
    
    n_sg = (2 ** max_depth) - 1
    im_height = max_depth * res
    im_width = n_sg * res
    im = 0.7 * np.ones((n_logged_samples, 3, im_height, im_width))
    
    if 'gt_match_dists' in model_output:
        usage_prob = tree.df.match_dist.sum(2)
    elif 'existence_predictor' in model_output:
        usage_prob = model_output.existence_predictor.existence
    else:
        usage_prob = None
    usage_prob = tensor2np(usage_prob, n_logged_samples)
    
    for n, node in enumerate(tree.depth_first_iter()):
        level = max_depth - node.depth
        # Handle 5-dimensional masks
        im_batch = imgtensor2np(node.subgoal[elem][:, [0, 1, -1]], n_logged_samples)
        if usage_prob is not None:
            imgs = draw_frame(im_batch, usage_prob[:, n])
            im_batch = np.stack(imgs, 0)
        im[:, :, level*res : (level+1)*res, n*res : (n+1)*res] = im_batch
    return im


def plot_balanced_tree_with_actions(model_output, inputs, n_logged_samples, get_prob_fcn=None):
    tree = model_output.tree
    batch, channels, res, _ = tree.subgoal.images.shape
    _, action_dim = tree.subgoal.a_l.shape
    max_depth = tree.depth
    n_sg = (2 ** max_depth) - 1

    im_height = (max_depth * 2 + 1) * res  # plot all actions (*2) and start/end frame again (+1)
    im_width = (n_sg + 2) * res
    im = np.asarray(0.7 * np.ones((n_logged_samples, im_height, im_width, 3)), dtype=np.float32)

    # insert start and goal frame
    if inputs is not None:
        im[:, :res, :res] = imgtensor2np(inputs.traj_seq[:n_logged_samples, 0], n_logged_samples).transpose(0, 2, 3, 1)
        im[:, :res, -res:] = imgtensor2np(batchwise_index(inputs.traj_seq[:n_logged_samples], model_output.end_ind[:n_logged_samples]),
                                          n_logged_samples).transpose(0, 2, 3, 1)

    if 'norm_gt_action_match_dists' in model_output:
        action_usage_prob = np.max(tensor2np(model_output.norm_gt_action_match_dists, n_logged_samples), axis=2)

    step = 1
    for i, segment in enumerate(tree):
        level = 2 * (max_depth - segment.depth + 1)
        dx = 2 ** (segment.depth - 2)
        im[:, level * res : (level + 1) * res, step * res: (step + 1) * res] = \
            imgtensor2np(segment.subgoal.images[:n_logged_samples], n_logged_samples).transpose(0, 2, 3, 1)
        a_l, a_r = tensor2np(segment.subgoal.a_l, n_logged_samples), tensor2np(segment.subgoal.a_r, n_logged_samples)
        if get_prob_fcn is not None:
            usage_prob_l, usage_prob_r = get_prob_fcn(segment)
        else:
            usage_prob_l, usage_prob_r = action_usage_prob[:, 2*i], action_usage_prob[:, 2*i+1]
        for b in range(n_logged_samples):
            im[b, (level-1) * res : level * res, int((step-dx) * res): int((step - dx + 1) * res)] = \
                framed_action2img(a_l[b], usage_prob_l[b], res, channels)
            im[b, (level - 1) * res: level * res, int((step + dx) * res): int((step + dx + 1) * res)] = \
                framed_action2img(a_r[b], usage_prob_r[b], res, channels)
        step += 1
    return im


def plot_pruned_tree(tree, check_pruned_fcn=lambda x, b: x.pruned[b], plot_matched=False):
    """Plots subgoal tree, but only non-pruned nodes.
       'check_pruned_fcn' allows flexible definition of what 'pruned' means
       'plot_matched': if True, plot nodes at positions where they were matched
    """
    max_depth = tree.depth
    batch, channels, res, _ = tree.subgoal.images.shape
    n_sg = (2 ** max_depth) - 1
    im_height = max_depth * res
    im_width = n_sg * res
    im = 1.0 * torch.ones((batch, channels, im_height, im_width))
    step = 0
    for segment in tree:
        level = max_depth - segment.depth
        for b in range(batch):
            if check_pruned_fcn(segment, b): continue      # only plot non-pruned elements of the tree
            pos = segment.match_eval_idx[b] if plot_matched else step
            im[b, :, level * res: (level + 1) * res, pos * res: (pos + 1) * res] = (segment.subgoal.images[b]+1 / 2)
        step += 1
    return im


def plot_val_tree(model_output, inputs, n_logged_samples=3):
    tree = model_output.tree
    batch, _, channels, res, _ = tree.subgoals.images.shape
    max_depth = tree.depth
    n_sg = (2 ** max_depth) - 1

    dpi = 10
    fig_height, fig_width = 2 * res, n_sg * res

    im_height, im_width = max_depth*res + fig_height, 2*res + fig_width
    im = np.asarray(0.7 * np.ones((n_logged_samples, im_height, im_width, 3)), dtype=np.float32)

    # plot existence probabilities
    if 'p_n_hat' in tree.subgoals:
        p_n_hat = tensor2np(tree.df.p_n_hat, n_logged_samples)
        for i in range(n_logged_samples):
            im[i, :res, res:-res] = plot_dists([p_n_hat[i]], res, fig_width, dpi)
    if 'p_a_l_hat' in tree.subgoals:
        p_a_hat = tensor2np(sort_actions_depth_first(model_output, ['p_a_l_hat', 'p_a_r_hat'], n_logged_samples))
        for i in range(n_logged_samples):
            im[i, res:2*res, int(3*res/4):int(-3*res/4)] = plot_dists([p_a_hat[i]], res, fig_width + int(res/2), dpi)
        im = np.concatenate((im[:, :fig_height],
                             plot_balanced_tree_with_actions(model_output, inputs, n_logged_samples,
                                                             get_prob_fcn=lambda s: (tensor2np(s.subgoal.p_a_l_hat),
                                                                                     tensor2np(s.subgoal.p_a_r_hat)))), axis=1)
    else:
        with param(n_logged_samples=n_logged_samples):
            im[:, fig_height:, res:-res] = plot_balanced_tree(model_output).transpose((0, 2, 3, 1))

    # insert start and goal frame
    if inputs is not None:
        im[:, :res, :res] = imgtensor2np(inputs.traj_seq[:n_logged_samples, 0], n_logged_samples).transpose(0, 2, 3, 1)
        im[:, :res, -res:] = imgtensor2np(batchwise_index(inputs.traj_seq[:n_logged_samples], model_output.end_ind[:n_logged_samples]),
                                          n_logged_samples).transpose(0, 2, 3, 1)

    return im


def plot_hierarchical_match_dists(tree, n_logged_samples=3):
    """Plots tree-overview of matching distributions."""
    fig_height, fig_width = 300, 300
    dpi = 10

    max_depth = tree.depth
    im_height = fig_height * max_depth
    im_width = fig_width * (2**(max_depth-1))

    im = np.asarray(0.7 * np.ones((n_logged_samples, im_height, im_width, 3)), dtype=np.float32)
    dw = int(fig_width/2)

    ow = 0
    for segment in tree:
        oh = (max_depth - segment.depth) * fig_height
        c_n = tensor2np(segment.subgoal.c_n, n_logged_samples)
        c_n_prime = tensor2np(segment.subgoal.c_n_prime, n_logged_samples)
        for i in range(n_logged_samples):
            im[i, oh:oh+fig_height, ow:ow+fig_width] = plot_dists([c_n[i], c_n_prime[i]], fig_height, fig_width, dpi, linewidth=15.0)
        ow += dw    # in depth-first iteration every new plot is half a plot-size to the right

    return im


def plot_gt_matching_overview(model_output, inputs, plot_attr='match_dist'):
    """Plots overview of which predicted frames contributed to which subgoals."""
    if len(inputs.traj_seq_images.shape) > 2:
        assert inputs.traj_seq_images.shape[3] == inputs.traj_seq_images.shape[4]     # code can only handle square images atm
    batch, n_gt, channels, res, _ = inputs.traj_seq_images.shape
    n_logged_samples = PARAMS.n_logged_samples
    assert batch >= n_logged_samples
    tree = model_output.tree
    max_depth = tree.depth
    n_sg = (2**max_depth) - 1

    im_height = (n_gt+max_depth) * res
    im_width = (n_sg + 2) * res
    im = np.asarray(0.7 * np.ones((n_logged_samples, im_height, im_width, 3)), dtype=np.float32)

    # insert ground truth images on the left, soft estimates on the right, top to bottom
    # insert raw subgoal predictions tree at the bottom, left to right in depth-first order
    get_strip = lambda x, gt=False: imgtensor2np(x, n_logged_samples, gt).transpose(0, 1, 3, 4, 2)\
        .reshape(n_logged_samples, res * n_gt, res, channels)
    
    im[:, :res*n_gt, :res] = get_strip(inputs.traj_seq_images, gt=True)
    if 'soft_matched_estimates' in model_output:
        im[:, :res * n_gt, res:2*res] = get_strip(model_output.soft_matched_estimates)
    
    with param(n_logged_samples=n_logged_samples):
        im[:, -max_depth*res:, 2*res:] = plot_balanced_tree(model_output).transpose((0, 2, 3, 1))

    fig_height, fig_width = res, n_sg * res
    dpi = 10
    match_dists = tensor2np(tree.get_attr_df(plot_attr), n_logged_samples)
    for i in range(n_gt):
        for b in range(n_logged_samples):
            match_plot = plot_dists([match_dists[b, :, i]], fig_height, fig_width, dpi, linewidth=3.0)
            im[b, i*res : (i+1)*res, 2*res:] = match_plot

    return im


def plot_pruned_seqs(model_output, inputs, n_logged_samples=3, max_seq_len=None):
    """Plots the pruned output sequences of the SH-Pred model."""
    assert "images" in model_output.dense_rec       # need pruned predicted images of SH-Pred model
    if inputs is not None:
        batch, n_gt_imgs, channels, res, _ = inputs.traj_seq.shape
    else:
        batch = len(model_output.dense_rec.images)
        assert batch == 1      # con currently only handle batch size 1
        n_gt_imgs, channels, res, _ = model_output.dense_rec.images[0].shape
    MAX_SEQ_LEN = int(n_gt_imgs * 1.5) if not max_seq_len else max_seq_len

    im_height = 2 * res
    im_width = (MAX_SEQ_LEN+1) * res
    im = np.asarray(0.7 * np.ones((n_logged_samples, im_height, im_width, 3)), dtype=np.float32)
    pred_imgs = list(map(imgtensor2np, model_output.dense_rec.images[:n_logged_samples]))
    max_len = min(n_gt_imgs, MAX_SEQ_LEN)
    for b in range(n_logged_samples):
        if pred_imgs[b] is None: continue
        seq_len = min(pred_imgs[b].shape[0], MAX_SEQ_LEN)
        max_len = max(max_len, seq_len)
        im[b, -res:, res:(seq_len+1)*res] = pred_imgs[b][:seq_len].transpose(2, 0, 3, 1).reshape(res, seq_len*res, channels)
    if inputs is not None:
        im[:, :res, :(n_gt_imgs*res)] = imgtensor2np(inputs.traj_seq, n_logged_samples).transpose(0, 3, 1, 4, 2)\
                                            .reshape(n_logged_samples, res, n_gt_imgs*res, channels)
    if "actions" in model_output.dense_rec \
            and model_output.dense_rec.actions is not None \
            and (True in [a is not None for a in model_output.dense_rec.actions]) \
            and (inputs is None or inputs.actions.shape[-1] == 2):
        ac_im = np.asarray(0.7 * np.ones((n_logged_samples, res, im_width, 3)), dtype=np.float32)
        pred_ac = list(map(tensor2np, model_output.dense_rec.actions[:n_logged_samples]))
        for b in range(n_logged_samples):
            if pred_ac[b] is None: continue
            seq_len = min(pred_ac[b].shape[0], MAX_SEQ_LEN)
            ac_im[b, :, :seq_len*res] = batch_action2img(pred_ac[b][None, :seq_len], res, channels).transpose(0, 2, 1, 3, 4)\
                .reshape(res, seq_len*res, channels)
        im = np.concatenate((im, ac_im), axis=1)

    # prepare GIF version
    gif_imgs = np.swapaxes(im.reshape(n_logged_samples, im.shape[1], MAX_SEQ_LEN+1, res, channels), 0, 2)[:max_len+1] \
                    .reshape(max_len+1, im.shape[1], res * n_logged_samples, channels)

    return im, gif_imgs


def plot_traj_following_overview(traj_seq, exec_seq, goal):
    """Plots overview of pruned subplans.
    :arg traj_seq: sequence of data frames, [N, H, W, C]
    :arg exec_seq: sequence of executed frames, [N, H, W, C]
    :arg goal: goal image, [H, W, C]
    """
    seq_len, res, _, channels = exec_seq.shape
    exec_seq = exec_seq / 255.0 if np.max(exec_seq) > 1.0 else exec_seq
    MAX_SEQ_LEN = int(seq_len * 1.5)

    im_height = 2*res
    im_width =  (MAX_SEQ_LEN + 1) * res
    im = 0.7 * np.ones((im_height, im_width, channels), dtype=np.float32)
    goal_separator = np.ones((res, 4, channels))

    # add executed sequence at top
    im[:res, :seq_len*res] = exec_seq.transpose(1, 0, 2, 3).reshape(res, seq_len * res, channels)
    im[:res, seq_len*res:(seq_len+1)*res] = goal
    im[:res, seq_len*res-2:seq_len*res+2] = goal_separator

    traj_len = traj_seq.shape[0]
    im[res: res*2, :traj_len*res] = traj_seq.transpose(1, 0, 2, 3).reshape(res, traj_len* res, channels)

    return np.expand_dims(im, axis=0)


def plot_planning_overview(planner_outputs, exec_seq, goal):
    """Plots overview of pruned subplans.
    :arg planner_outputs: list of tuples (step, model_output)
    :arg exec_seq: sequence of executed frames, [N, H, W, C]
    :arg goal: goal image, [H, W, C]
    """
    seq_len, res, _, channels = exec_seq.shape
    exec_seq = exec_seq / 255.0 if np.max(exec_seq) > 1.0 else exec_seq
    MAX_SEQ_LEN = int(seq_len * 1.5)
    if "actions" in planner_outputs[0][1].dense_rec and planner_outputs[0][1].dense_rec.actions is not None:
        if (True in [a is not None for a in planner_outputs[0][1].dense_rec.actions]):
            seq_height = 2*res
        else:
            seq_height = res
    else:
        seq_height = res

    im_height = res + seq_height * len(planner_outputs)
    im_width = (planner_outputs[-1][0] + MAX_SEQ_LEN + 1) * res
    im = 0.7 * np.ones((im_height, im_width, channels), dtype=np.float32)
    goal_separator = np.ones((res, 4, channels))

    # add executed sequence at top
    im[:res, :seq_len*res] = exec_seq.transpose(1, 0, 2, 3).reshape(res, seq_len * res, channels)
    im[:res, seq_len*res:(seq_len+1)*res] = goal
    im[:res, seq_len*res-2:seq_len*res+2] = goal_separator

    # add planned pruned seqs
    for i, (plan_step, planner_output) in enumerate(planner_outputs):
        if isinstance(planner_output, list):
            pred_seq = planner_output
        else:
            pruned_seq, _ = plot_pruned_seqs(planner_output, None, n_logged_samples=1, max_seq_len=MAX_SEQ_LEN)
            pred_seq = pruned_seq[0, res:, res:]     # remove gt sequence row and empty first element of predicted row
        seq_width = pred_seq.shape[1]
        im[res+i*seq_height : res+(i+1)*seq_height, plan_step*res : plan_step*res + seq_width] = pred_seq
        im[res+i*seq_height : 2*res + i*seq_height, plan_step*res + seq_width : (plan_step+1)*res + seq_width] = goal
        im[res+i*seq_height : 2*res + i*seq_height, plan_step*res + seq_width - 2 : plan_step*res + seq_width+2] = goal_separator

    return np.expand_dims(im, axis=0)


def unstack(arr):
        arr = np.split(arr, arr.shape[0], 0)
        arr = [a.squeeze() for a in arr]
        return arr


def plot_inverse_model_actions(model_output, inputs, n_logged_samples=5):
    #assert inputs.actions.shape[-1] == 2  # code can only handle 2-dim actions
    batch, n_gt_imgs, channels, res, _ = inputs.traj_seq.shape

    def make_row(arr):
        """stack images in a row along batch dimension"""
        return np.concatenate(unstack(arr), 1)

    if len(model_output.action_targets.shape) <= 2:
        action_targets, actions = model_output.action_targets, model_output.actions
    else:
        action_targets, actions = model_output.action_targets[:, 0], model_output.actions[:, 0]
        model_output.img_t0, model_output.img_t1 = inputs.traj_seq[:, 0], inputs.traj_seq[:, 1]
    if action_targets.shape[-1] > 2:
        actions, action_targets = actions[..., :2], action_targets[..., :2]

    input_action_imgs = batch_action2img(tensor2np(action_targets[:, None], n_logged_samples), res, channels)
    pred_action_imgs = batch_action2img(tensor2np(actions[:, None], n_logged_samples), res, channels)

    image_rows = []
    image_rows.append(np.transpose(tensor2np(model_output.img_t0, n_logged_samples), [0, 2, 3, 1]))
    image_rows.append(np.transpose(tensor2np(model_output.img_t1, n_logged_samples), [0, 2, 3, 1]))
    image_rows.append(input_action_imgs.squeeze())
    image_rows.append(pred_action_imgs.squeeze())

    image_rows = [make_row(item) for item in image_rows]

    im = (np.concatenate(image_rows, 0)[None] + 1.0)/2

    return im


def plot_policy_actions(policy_actions_raw, input_actions_raw, input_frames, n_logged_samples=5):
    #assert input_actions.shape[-1] == 2  # code can only handle 2-dim actions
    input_actions = input_actions_raw[..., :2]
    policy_actions = policy_actions_raw[..., :2]
    batch, n_gt_imgs, channels, res, _ = input_frames.shape

    input_action_imgs = batch_action2img(tensor2np(input_actions, n_logged_samples), res, channels).transpose(0, 2, 1, 3, 4).reshape(n_logged_samples, res, res * (n_gt_imgs-1), channels)
    pred_action_imgs = batch_action2img(tensor2np(policy_actions, n_logged_samples), res, channels).transpose(0, 2, 1, 3, 4).reshape(n_logged_samples, res, res * (n_gt_imgs-1), channels)

    input_images = tensor2np(input_frames[:, :-1], n_logged_samples).transpose(0, 3, 1, 4, 2) \
                                                .reshape(n_logged_samples, res, res * (n_gt_imgs-1), channels)

    im = (np.concatenate((input_images, input_action_imgs, pred_action_imgs), axis=1) + 1.0) / 2
    return im


def make_gif(seqs, n_seqs_logged=5):
    """Fuse sequences in list + bring in gif format. Uses the imgtensor2np function"""
    seqs = [pad_sequence(seq, batch_first=True) for seq in seqs]
    seqs = [imgtensor2np(s, n_logged_samples=n_seqs_logged) for s in seqs]
    stacked_seqs = seqs[0]
    
    if len(seqs) > 1:
        padding = np.zeros_like(stacked_seqs)[:, :, :, :2]
        padded_seqs = list(np.concatenate([padding, seq], axis=3) for seq in seqs[1:])
        stacked_seqs = np.concatenate([stacked_seqs] + padded_seqs, axis=3)
    
    batch, time, channels, height, width = stacked_seqs.shape
    return stacked_seqs.transpose(1, 2, 3, 0, 4).reshape(time, channels, height, width * batch)


PREV_OBJS = None
def eval_mem_objs():
    """A simple helper function to evaluate the number of objects currently in memory (CPU or GPU) and print the
       difference to the objects in memory when previously calling this function."""
    import gc
    gc.collect()
    param_size, tensor_size = 0, 0
    objs = dict()
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                # print(type(obj), obj.size())
                if isinstance(obj, torch.nn.parameter.Parameter):
                    param_size = param_size + 1
                else:
                    tensor_size = tensor_size + 1
                    key = tuple(obj.size())
                    if key in objs:
                        objs[key] = objs[key] + 1
                    else:
                        objs[key] = 1
        except:
            pass
    print("#Params: {}".format(param_size))
    print("#Tensors: {}".format(tensor_size))

    global PREV_OBJS
    if PREV_OBJS is not None:
        diff = dict()
        for key in objs:
            if key in PREV_OBJS:
                d = objs[key] - PREV_OBJS[key]
                if d != 0:
                    diff[key] = d
            else:
                diff[key] = objs[key]

        import pprint
        pprint.pprint(diff)

    PREV_OBJS = objs

