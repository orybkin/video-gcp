"""based on: https://github.com/google-research/google-research/blob/master/frechet_video_distance/example.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import tensorflow as tf

from blox import AttrDict
from gcp.evaluation.fvd import frechet_video_distance as fvd

# Number of videos must be divisible by 16.
NUMBER_OF_VIDEOS = 16


def get_trainer_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", help="path to ground truth sequence .npy file")
    parser.add_argument("--pred", help="path to predicted sequence .npy file")
    return parser.parse_args()


def main(argv):
    args = get_trainer_args()

    def load_videos(path):
        print("Loading trajectories from {}".format(path))
        if not path.endswith('.npy'): raise ValueError("Can only read in .npy files!")
        seqs = (np.load(path).transpose(0, 1, 3, 4, 2) + 1) / 2
        assert len(seqs.shape) == 5     # need [batch, T, W, H, C] input data
        assert seqs.shape[-1] == 3      # assume 3-channeled seq with channel in last dim
        if seqs.max() <= 1: seqs *= 255     # assume [0...255] range
        return seqs

    gt_seqs = load_videos(args.gt)
    pred_seqs = load_videos(args.pred)

    assert gt_seqs.shape == pred_seqs.shape

    batch, time, h, w, c = gt_seqs.shape
    n_batches = int(np.floor(batch / NUMBER_OF_VIDEOS))     # needs to be dividable by NUMBER_OF_VIDEOS
    print("Evaluating batch of {} sequences of shape {}...".format(NUMBER_OF_VIDEOS, (time, h, w, c)))

    # Step 1: collect all embeddings (needs to run in loop bc network can only handle batch_size 16)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    end_ind = np.argmax(np.all(np.abs(gt_seqs) < 1e-6, axis=(2, 3, 4)), axis=-1)  # check for black images
    end_ind[end_ind == 0] = time
    embeddings = AttrDict()
    for key, seq in [['gt', gt_seqs], ['pred', pred_seqs]]:
        stored_embeds = []
        for i, s in enumerate(seq):
            length = end_ind[i]
            if length < 10: continue
            with tf.Graph().as_default():
                # construct embedding graph
                seq_ph = tf.placeholder(dtype=tf.float32, shape=(1, length, h, w, c))
                embed = fvd.create_id3_embedding(fvd.preprocess(seq_ph, (224, 224)))
                with tf.Session(config=config) as sess:
                    sess.run(tf.global_variables_initializer())
                    sess.run(tf.tables_initializer())
                    feed_dict = {seq_ph: s[:length][None]}
                    #feed_dict = {seq_ph: np.concatenate((s[:length][None], np.zeros([NUMBER_OF_VIDEOS-1, length] + s.shape[1:])))}
                    print("{} - Seq {} - Length: {}".format(key, i, length))
                    e = sess.run(embed, feed_dict=feed_dict)
                stored_embeds.append(e[0])
        embeddings[key] = np.stack(stored_embeds)
    print("Generated embeddings!")

    # Step 2: evaluate the FVD
    with tf.Graph().as_default():
        gt_embed_ph = tf.placeholder(dtype=tf.float32, shape=embeddings.gt.shape)
        pred_embed_ph = tf.placeholder(dtype=tf.float32, shape=embeddings.pred.shape)
        result = fvd.calculate_fvd(gt_embed_ph, pred_embed_ph)
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            feed_dict = {gt_embed_ph: embeddings.gt, pred_embed_ph: embeddings.pred}
            print("FVD is: %.2f." % sess.run(result, feed_dict=feed_dict))


if __name__ == "__main__":
    tf.app.run(main)
