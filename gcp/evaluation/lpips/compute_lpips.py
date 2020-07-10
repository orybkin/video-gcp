import argparse
import numpy as np
import torch

import models
from blox import batch_apply
from blox.basic_types import map_dict

def get_trainer_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", help="path to ground truth sequence .npy file")
    parser.add_argument("--pred", help="path to predicted sequence .npy file")
    parser.add_argument("--batch_size", default=8, type=int, help="batch size for network forward pass")
    parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')
    return parser.parse_args()


def main():
    args = get_trainer_args()

    def load_videos(path):
        print("Loading trajectories from {}".format(path))
        if not path.endswith('.npy'): raise ValueError("Can only read in .npy files!")
        seqs = np.load(path)
        assert len(seqs.shape) == 5     # need [batch, T, C, H, W] input data
        assert seqs.shape[2] == 3      # assume 3-channeled seq with channel in last dim
        seqs = torch.Tensor(seqs)
        if args.use_gpu: seqs = seqs.cuda()
        return seqs     # range [-1, 1]

    gt_seqs = load_videos(args.gt)
    pred_seqs = load_videos(args.pred)
    print('shape: ', gt_seqs.shape)

    assert gt_seqs.shape == pred_seqs.shape

    n_seqs, time, c, h, w = gt_seqs.shape
    n_batches = int(np.floor(n_seqs / args.batch_size))

    # import pdb; pdb.set_trace()
    # get sequence mask (for sequences with variable length
    mask = 1 - torch.all(torch.all(torch.all((gt_seqs + 1.0).abs() < 1e-6, dim=-1), dim=-1), dim=-1)  # check for black images
    mask2 = 1 - torch.all(torch.all(torch.all((gt_seqs).abs() < 1e-6, dim=-1), dim=-1), dim=-1)  # check for gray images
    mask = mask * mask2

    # Initializing the model
    model = models.PerceptualLoss(model='net-lin', net='alex', use_gpu=args.use_gpu)

    # run forward pass to compute LPIPS distances
    distances = []
    for b in range(n_batches):
        x, y = gt_seqs[b*args.batch_size : (b+1)*args.batch_size], pred_seqs[b*args.batch_size : (b+1)*args.batch_size]
        lpips_dist = batch_apply(model, x, y)
        distances.append(lpips_dist)
    distances = torch.cat(distances)
    mean_distance = distances[mask].mean()

    print("LPIPS distance: {}".format(mean_distance))


if __name__ == "__main__":
    main()
