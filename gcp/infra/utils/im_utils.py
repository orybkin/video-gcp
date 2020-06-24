import cv2
import moviepy.editor as mpy
import os
import numpy as np
import matplotlib.pyplot as plt
import pdb


def resize_store(t, target_array, input_array):
    target_img_height, target_img_width = target_array.shape[2:4]
    assert input_array.dtype == np.uint8

    if len(input_array.shape) == 3: # if image has only one channel tile last dimension 3 times
        input_array = input_array[..., None].astype(np.float32)
        input_array += abs(np.min(input_array))  # make sure no negative values
        input_array = color_code(input_array[0], renormalize=True)[None]  # take first camera from input and make colormap
        input_array = (input_array*255).astype(np.uint8)

    # plt.switch_backend('TkAgg')
    # plt.imshow(input_array[0])
    # plt.show()

    if (target_img_height, target_img_width) == input_array.shape[1:3]:
        for i in range(input_array.shape[0]):
            target_array[t, i] = input_array[i]
    else:
        for i in range(input_array.shape[0]):   # loop over cameras
            target_array[t, i] = cv2.resize(input_array[i], (target_img_width, target_img_height),
                                            interpolation=cv2.INTER_AREA)

def color_code(inp, renormalize=False):
    cmap = plt.cm.get_cmap('viridis')
    if renormalize:
        inp /= (np.max(inp)+1e-6)
    colored_distrib = cmap(np.squeeze(inp))[:, :, :3]
    return colored_distrib


def npy_to_gif(im_list, filename, fps=5):
    save_dir = '/'.join(str.split(filename, '/')[:-1])

    if not os.path.exists(save_dir):
        print('creating directory: ', save_dir)
        os.makedirs(save_dir)

    if isinstance(im_list, np.ndarray):
        im_list = list(im_list)

    clip = mpy.ImageSequenceClip(im_list, fps=fps)
    clip.write_gif(filename + '.gif')


def npy_to_mp4(im_list, filename, fps=4):
    save_dir = '/'.join(str.split(filename, '/')[:-1])

    if not os.path.exists(save_dir):
        print('creating directory: ', save_dir)
        os.mkdir(save_dir)

    clip = mpy.ImageSequenceClip(im_list, fps=fps)
    clip.write_videofile(filename + '.mp4')
