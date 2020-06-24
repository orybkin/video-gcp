import matplotlib; matplotlib.use('Agg')
import pickle as pkl
import numpy as np
import argparse

from gcp.infra.envs.miniworld_env.utils.multiroom2d_layout import define_layout
from gcp.infra.policy.prm_policy.prm_policy import plan_room_seq


def n_room_path(start, end, layout):
    return len(plan_room_seq(start, end, layout.doors))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="path to the config file directory")
    parser.add_argument('--n_rooms', default=9, type=int,
                        help='number of rooms in navigation layout')
    parser.add_argument('--n_tasks', default=100, type=int,
                        help='number of task instances in eval set')
    parser.add_argument('--max_seq_len', default=200, type=int,
                        help='maximum length of sequence (for cost computation')
    return parser.parse_args()


def main():
    args = parse_args()
    FILE = args.path

    rooms_per_side = int(np.sqrt(args.n_rooms))
    layout = define_layout(rooms_per_side, None)

    with open(FILE, 'rb') as pickle_file:
        data = pkl.load(pickle_file)

    paths = data['full_traj']

    success, rooms_to_goal, rooms_traversed = 0, [], []
    penalized_length = 0.

    for i in range(args.n_tasks):
        # extract start / final / goal position and room
        goal_pos = data['reset_state'][i]['goal'][-2:] / 27
        final_pos = paths[i][-1][:2] / 27
        start_pos = paths[i][0][:2] / 27
        goal_pos[1] *= -1; final_pos[1] *= -1; start_pos[1] *= -1
        goal_room = layout.coords2ridx(goal_pos[0], goal_pos[1])
        final_room = layout.coords2ridx(final_pos[0], final_pos[1])
        start_room = layout.coords2ridx(start_pos[0], start_pos[1])

        # compute success
        if final_room == goal_room:
            success += 1

        # compute length
        path = np.stack([p[:2] for p in paths[i]])
        path_len = np.sum(np.linalg.norm(path[1:] - path[:-1], axis=-1))
        penalized_length += path_len if final_room == goal_room else args.max_seq_len

        # compute number of rooms to goal / traversed
        rooms_to_goal += [n_room_path(final_room, goal_room, layout)]
        rooms_traversed += [n_room_path(start_room, final_room, layout)]

    print("Success: \t{}".format(success / args.n_tasks))
    print("Cost: \t{:.2f}".format(penalized_length / args.n_tasks))

    print("")
    print("Room2Goal: \t{}\t{}".format(np.mean(rooms_to_goal), np.std(rooms_to_goal)))
    print("RTravers: \t{}\t{}".format(np.mean(rooms_traversed), np.std(rooms_traversed)))


if __name__ == "__main__":
    main()
