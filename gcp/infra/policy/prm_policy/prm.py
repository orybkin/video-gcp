"""

Probablistic Road Map (PRM) Planner

author: Atsushi Sakai (@Atsushi_twi)
from: https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathPlanning/ProbabilisticRoadMap/probabilistic_road_map.py

"""

import math
import random

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
from blox import AttrDict

# parameter
PARAMS = AttrDict(
    n_samples=3000,  # number of sample_points
    n_knn=30, # number of edge from one sampled point
    max_edge_len=0.1,  # [m] Maximum edge length
    cost_fcn=lambda d: d,   # cost function for PRM planning, input is distance between nodes
)


class Node:
    """
    Node class for dijkstra search
    """

    def __init__(self, x, y, cost, pind):
        self.x = x
        self.y = y
        self.cost = cost
        self.pind = pind

    def __str__(self):
        return str(self.x) + "," + str(self.y) + "," + str(self.cost) + "," + str(self.pind)


class KDTree:
    """
    Nearest neighbor search class with KDTree
    """

    def __init__(self, data):
        # store kd-tree
        self.tree = scipy.spatial.cKDTree(data)

    def search(self, inp, k=1):
        """
        Search NN

        inp: input data, single frame or multi frame

        """

        if len(inp.shape) >= 2:  # multi input
            index = []
            dist = []

            for i in inp.T:
                idist, iindex = self.tree.query(i, k=k)
                index.append(iindex)
                dist.append(idist)

            return index, dist

        dist, index = self.tree.query(inp, k=k)
        return index, dist

    def search_in_distance(self, inp, r):
        """
        find points with in a distance r
        """

        index = self.tree.query_ball_point(inp, r)
        return index


def PRM_planning(sx, sy, gx, gy, ox, oy, rr, params, n_samples, sampled_points=None):

    obkdtree = KDTree(np.vstack((ox, oy)).T)

    sample_x, sample_y = sample_points(sx, sy, gx, gy, rr, ox, oy, obkdtree, n_samples, sampled_points)

    road_map = generate_roadmap(sample_x, sample_y, rr, obkdtree, params)

    rx, ry, success = dijkstra_planning(
        sx, sy, gx, gy, ox, oy, rr, road_map, sample_x, sample_y, params.cost_fcn)

    return rx[::-1], ry[::-1], success


def is_collision(sx, sy, gx, gy, rr, okdtree, params):
    x = sx
    y = sy
    dx = gx - sx
    dy = gy - sy
    yaw = math.atan2(gy - sy, gx - sx)
    d = math.sqrt(dx**2 + dy**2)

    if d >= params.max_edge_len:
        return True

    D = rr
    nstep = round(d / D)

    for i in range(nstep):
        idxs, dist = okdtree.search(np.array([x, y]).reshape(2, 1))
        if dist[0] <= rr:
            return True  # collision
        x += D * math.cos(yaw)
        y += D * math.sin(yaw)

    # goal point check
    idxs, dist = okdtree.search(np.array([gx, gy]).reshape(2, 1))
    if dist[0] <= rr:
        return True  # collision

    return False  # OK


def generate_roadmap(sample_x, sample_y, rr, obkdtree, params):
    """
    Road map generation

    sample_x: [m] x positions of sampled points
    sample_y: [m] y positions of sampled points
    rr: Robot Radius[m]
    obkdtree: KDTree object of obstacles
    """

    road_map = []
    nsample = len(sample_x)
    skdtree = KDTree(np.vstack((sample_x, sample_y)).T)

    for (i, ix, iy) in zip(range(nsample), sample_x, sample_y):

        index, dists = skdtree.search(
            np.array([ix, iy]).reshape(2, 1), k=nsample)
        inds = index[0]
        edge_id = []
        #  print(index)

        for ii in range(1, len(inds)):
            nx = sample_x[inds[ii]]
            ny = sample_y[inds[ii]]

            if not is_collision(ix, iy, nx, ny, rr, obkdtree, params):
                edge_id.append(inds[ii])

            if len(edge_id) >= params.n_knn:
                break

        road_map.append(edge_id)

    return road_map


def dijkstra_planning(sx, sy, gx, gy, ox, oy, rr, road_map, sample_x, sample_y, cost_fcn):
    """
    gx: goal x position [m]
    gx: goal x position [m]
    ox: x position list of Obstacles [m]
    oy: y position list of Obstacles [m]
    reso: grid resolution [m]
    rr: robot radius[m]
    """

    nstart = Node(sx, sy, 0.0, -1)
    ngoal = Node(gx, gy, 0.0, -1)

    openset, closedset = dict(), dict()
    openset[len(road_map) - 2] = nstart

    success = True
    while True:
        if not openset:
            print("Cannot find path")
            success = False
            break

        c_id = min(openset, key=lambda o: openset[o].cost)
        current = openset[c_id]

        if c_id == (len(road_map) - 1):
            ngoal.pind = current.pind
            ngoal.cost = current.cost
            break

        # Remove the item from the open set
        del openset[c_id]
        # Add it to the closed set
        closedset[c_id] = current

        # expand search grid based on motion model
        for i in range(len(road_map[c_id])):
            n_id = road_map[c_id][i]
            dx = sample_x[n_id] - current.x
            dy = sample_y[n_id] - current.y
            d = math.sqrt(dx**2 + dy**2)
            node = Node(sample_x[n_id], sample_y[n_id],
                        current.cost + cost_fcn(d), c_id)

            if n_id in closedset:
                continue
            # Otherwise if it is already in the open set
            if n_id in openset:
                if openset[n_id].cost > node.cost:
                    openset[n_id].cost = node.cost
                    openset[n_id].pind = c_id
            else:
                openset[n_id] = node

    # generate final course
    rx, ry = [ngoal.x], [ngoal.y]
    pind = ngoal.pind
    while pind != -1:
        n = closedset[pind]
        rx.append(n.x)
        ry.append(n.y)
        pind = n.pind

    return rx, ry, success


def sample_points(sx, sy, gx, gy, rr, ox, oy, obkdtree, n_samples, sampled_points=None):
    maxx = max(ox)
    maxy = max(oy)
    minx = min(ox)
    miny = min(oy)

    sample_x, sample_y = [], []

    def maybe_add_pt(tx, ty):
        index, dist = obkdtree.search(np.array([tx, ty]).reshape(2, 1))
        if dist[0] >= rr:
            sample_x.append(tx)
            sample_y.append(ty)

    if sampled_points is None:
        while len(sample_x) <= n_samples:
            tx = (random.random() - (maxx - minx)/2) * (maxx - minx)
            ty = (random.random() - (maxy - miny)/2) * (maxy - miny)
            maybe_add_pt(tx, ty)
    else:
        for tx, ty in zip(*sampled_points):
            maybe_add_pt(tx, ty)

    sample_x.append(sx)
    sample_y.append(sy)
    sample_x.append(gx)
    sample_y.append(gy)

    return sample_x, sample_y


def add_horizontal_line(x_range, y):
    ox = np.linspace(x_range[0], x_range[1], x_range[1]-x_range[0]+1)
    oy = y * np.ones_like(ox)
    return np.stack([ox, oy], axis=0)


def add_vertical_line(y_range, x):
    oy = np.linspace(y_range[0], y_range[1], y_range[1]-y_range[0]+1)
    ox = x * np.ones_like(oy)
    return np.stack([ox, oy], axis=0)



def main():
    from gcp.infra.envs.miniworld_env.utils.multiroom2d_layout import define_layout
    from gcp.infra.envs.miniworld_env.utils.sampling_fcns import RoomSampler2d
    print(__file__ + " start!!")
    show_animation = True
    ROOMS_PER_SIDE = 3
    SAMPLE_PER_ROOM = 50

    layout = define_layout(ROOMS_PER_SIDE, texture_dir=None)
    state_sampler = RoomSampler2d(ROOMS_PER_SIDE)

    # sample start and goal
    s, g = state_sampler.sample(), state_sampler.sample()
    sx, sy, gx, gy = s[0], s[1], g[0], g[1]

    # sample PRM points
    px, py = [], []
    for _ in range(SAMPLE_PER_ROOM * ROOMS_PER_SIDE**2):
        p = state_sampler.sample()
        px.append(p[0]); py.append(p[1])
    samples = np.asarray([[state_sampler.sample_door(d[0], d[1]) for _ in range(3)]
                          for d in layout.doors]).transpose(2, 0, 1).reshape(2, -1)
    px.extend(samples[0]); py.extend(samples[1])

    ox, oy = layout.ox, layout.oy

    if show_animation:
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "^r")
        plt.plot(gx, gy, "^c")
        plt.grid(True)
        plt.axis("equal")

    rx, ry, success = PRM_planning(sx, sy, gx, gy, ox, oy, layout.robot_size, PARAMS,
                                   SAMPLE_PER_ROOM * ROOMS_PER_SIDE ** 2, [px, py])

    from scipy import interpolate
    tck, u = interpolate.splprep([rx, ry], s=0.0)
    rxs, rys = interpolate.splev(np.linspace(0, 1, int(len(rx)*2)), tck)

    assert success, 'Cannot found path'

    if show_animation:
        plt.plot(rx, ry, "-r")
        plt.plot(rxs, rys, "-*g")
        plt.show()


def vis_plan(px, py, layout):
    plt.plot(layout.ox, layout.oy, ".k")
    plt.grid(True)
    plt.axis("equal")
    plt.plot(px, py, "-r")
    plt.show()



if __name__ == '__main__':
    main()
