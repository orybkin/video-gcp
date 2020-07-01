import glob
import os
import random
from random import Random

import cv2
import numpy as np
from blox import AttrDict

# don't change these values, they need to correspond to the multiroom2d.xml file!
ROBOT_SIZE = 0.02
ROOM_SIZE = 1/3
DOOR_SIZE = 1.5 * 0.0667

MAZE_SEED = 42
MULTIMODAL = True
NON_SYMMETRIC = False #True


def define_layout_raw(rooms_per_side, _add_horizontal_line=None, _add_vertical_line=None):

    if _add_vertical_line is None:
        coord_offset = 0.5 * rooms_per_side * ROOM_SIZE     # center around (0,0)

        def _add_horizontal_line(x_range, y):
            ox = np.linspace(x_range[0], x_range[1], int((x_range[1] - x_range[0] + 1) * 100))
            oy = y * np.ones_like(ox)
            return np.stack([ox, oy], axis=0) - coord_offset

        def _add_vertical_line(y_range, x):
            oy = np.linspace(y_range[0], y_range[1], int((y_range[1] - y_range[0] + 1) * 100))
            ox = x * np.ones_like(oy)
            return np.stack([ox, oy], axis=0) - coord_offset

    # add outer boundaries
    table_size = ROOM_SIZE * rooms_per_side
    o = _add_horizontal_line([0, table_size], 0)
    o = np.concatenate((o, _add_horizontal_line([0, table_size], table_size)), axis=1)
    o = np.concatenate((o, _add_vertical_line([0, table_size], 0)), axis=1)
    o = np.concatenate((o, _add_vertical_line([0, table_size], table_size)), axis=1)

    # add wall segments
    rng = Random()
    rng.seed(MAZE_SEED)
    for wall_add_fcn in [_add_horizontal_line, _add_vertical_line]:
        for r in range(rooms_per_side):
            o = np.concatenate((o, wall_add_fcn([0, 1 * ROOM_SIZE/2 - DOOR_SIZE/2], (r+1) * ROOM_SIZE)), axis=1)
            for seg_idx in range(rooms_per_side - 1):
                if NON_SYMMETRIC and rng.random() < 0.1: continue
                o = np.concatenate((o, wall_add_fcn(
                    [(2*seg_idx+1) * ROOM_SIZE/2 + DOOR_SIZE/2, (2*(seg_idx+1)+1) * ROOM_SIZE/2 - DOOR_SIZE/2],
                    (r+1) * ROOM_SIZE)), axis=1)
            o = np.concatenate((o, wall_add_fcn([(rooms_per_side-0.5)*ROOM_SIZE + DOOR_SIZE/2, rooms_per_side*ROOM_SIZE],
                                                         (r + 1) * ROOM_SIZE)), axis=1)

    # generate maze and add doors
    doors = gen_doors_multimodal(rooms_per_side) if MULTIMODAL else generate_maze(rooms_per_side)
    for rx in range(rooms_per_side):
        for ry in range(rooms_per_side):
            if rx + 1 < rooms_per_side and \
                    (((rx, ry), (rx+1, ry)) not in doors and ((rx+1, ry), (rx, ry)) not in doors):
                door_center = ROOM_SIZE/2 + ry * ROOM_SIZE
                o = np.concatenate((o, _add_vertical_line([door_center - DOOR_SIZE/2, door_center + DOOR_SIZE/2],
                                   (rx + 1) * ROOM_SIZE)), axis=1)
            if ry + 1 < rooms_per_side and \
                    (((rx, ry), (rx, ry+1)) not in doors and ((rx, ry+1), (rx, ry)) not in doors):
                door_center = ROOM_SIZE/2 + rx * ROOM_SIZE
                o = np.concatenate((o, _add_horizontal_line([door_center - DOOR_SIZE/2, door_center + DOOR_SIZE/2],
                                   (ry + 1) * ROOM_SIZE)), axis=1)

    def coords2ridx(x, y):
        return x * rooms_per_side + (rooms_per_side-1) - y

    # translate to idx and make sure that smaller room idx comes first
    doors = [sorted((coords2ridx(d[0][0], d[0][1]), coords2ridx(d[1][0], d[1][1]))) for d in doors]

    return o, ROBOT_SIZE, table_size, doors


def generate_maze(rooms_per_side):
    """Returns a set of doors that, when open, generate a maze without shortcuts.
       Algorithm from here: https://github.com/maximecb/gym-miniworld"""
    doors = []
    rng = Random()
    rng.seed(MAZE_SEED)

    visited = []
    neighbors = [(0, 1), (1, 0), (-1, 0), (0, -1)]

    def visit(x, y):
        visited.append((x, y))
        rng.shuffle(neighbors)
        for dx, dy in neighbors.copy():
            nx, ny = x + dx, y + dy
            if nx < 0 or nx >= rooms_per_side or ny < 0 or ny >= rooms_per_side:
                continue    # not a valid neighbor
            if (nx, ny) in visited:
                continue    # neighbor already in visited states
            doors.append(((x, y), (nx, ny)))      # open door to neighbor
            visit(nx, ny)

    visit(0, 0)     # generate maze starting from room 0
    return doors


def gen_doors_multimodal(rooms_per_side):
    """Generates open layout with many doors that allows for multimodal trajectories."""
    # generate list of all doors
    doors = []
    neighbors = [(0, 1), (1, 0), (-1, 0), (0, -1)]

    def add_doors(x, y):
        for dx, dy in neighbors.copy():
            nx, ny = x + dx, y + dy
            if nx < 0 or nx >= rooms_per_side or ny < 0 or ny >= rooms_per_side:
                continue    # not a valid neighbor
            if ((x, y), (nx, ny)) in doors or ((nx, ny), (x, y)) in doors:
                continue
            doors.append(((x, y), (nx, ny)))
            add_doors(nx, ny)

    add_doors(0, 0)

    def maybe_remove(r1, r2):
        if (r1, r2) in doors:
            doors.remove((r1, r2))
        elif (r2, r1) in doors:
            doors.remove((r2, r1))

    # remove a few doors (i.e. add walls)
    if rooms_per_side == 3:     # add two walls around the middle room
        maybe_remove((1, 1), (1, 2))
        maybe_remove((1, 1), (1, 0))
    elif rooms_per_side == 4:
        maybe_remove((0, 3), (1, 3))
        maybe_remove((1, 0), (2, 0))
        maybe_remove((2, 1), (3, 1))
        maybe_remove((2, 2), (3, 2))
        maybe_remove((2, 3), (3, 3))
        maybe_remove((1, 3), (1, 4))
    elif rooms_per_side == 5:
        maybe_remove((0, 3), (1, 3))
        maybe_remove((1, 0), (2, 0))
        maybe_remove((2, 1), (3, 1))
        maybe_remove((2, 2), (3, 2))
        maybe_remove((2, 3), (3, 3))
        maybe_remove((1, 3), (1, 4))
        maybe_remove((1, 1), (1, 2))
        maybe_remove((2, 1), (2, 2))
        maybe_remove((3, 1), (3, 2))
        maybe_remove((4, 2), (4, 3))
    else:
        raise NotImplementedError
    return doors


def define_layout(rooms_per_side, texture_dir=None):
    if texture_dir is None:
        texture_dir = default_texture_dir()
    o, robot_size, table_size, doors = define_layout_raw(rooms_per_side)
    ox, oy = list(o[0]), list(o[1])

    def coords2ridx(x, y):
        """Translates float x, y coords into room index."""
        xy_room = [np.floor((c + table_size/2) / ROOM_SIZE) for c in [x, y]]
        ridx = xy_room[0] * rooms_per_side + (rooms_per_side-1) - xy_room[1]
        return int(ridx) if ridx.size == 1 else np.asarray(ridx, dtype=int)

    textures = load_textures(texture_dir) if texture_dir is not None else None

    return AttrDict(ox=ox, 
                    oy=oy, 
                    robot_size=robot_size,
                    table_size=table_size,
                    room_size=ROOM_SIZE,
                    door_size=DOOR_SIZE,
                    doors=doors,
                    coords2ridx=coords2ridx,
                    textures=textures,
                    texture_dir=texture_dir,
                    multimodal=MULTIMODAL,
                    non_symmetric=NON_SYMMETRIC,)


def default_texture_dir():
    texture_dir = None
    for dir in ['nav_9rooms', 'nav_25rooms']:
        path = os.path.join(os.environ["GCP_DATA_DIR"], dir, "textures")
        if os.path.exists(path):
            texture_dir = path
    assert texture_dir is not None   # need to download either 'nav_9room' or 'nav_25room' dataset to get textures
    return texture_dir


def load_textures(texture_dir):
    """Loads all textures from asset folder"""
    texture_files = glob.glob(os.path.join(texture_dir, "*.png"))
    texture_files = [os.path.basename(p) for p in texture_files]
    texture_files.sort()
    rng = random.Random()   # shuffle texture files
    rng.seed(42)
    rng.shuffle(texture_files)
    texture_files.remove("asphalt_1.png")   # this one is used for the floor
    # there was a bug in initial data collection, this is a hack to synchronize with the generated data
    # TODO remove the hardcoded textures when collecting new data
    HARDCODED_TEXTURE_FILES = ['floor_tiles_white.png', 'lg_style_01_4tile_d_result.png', 'lg_style_01_wall_blue_1.png',
                               'wood_1.png', 'lg_style_04_wall_cerise_d_result.png',
                               'lg_style_05_floor_blue_bright_d_result.png', 'cardboard_4.png',
                               'lg_style_03_wall_light_m_result.png',
                               'lg_style_02_wall_dblue_d_result.png',
                               'lg_style_02_wall_purple_d_result.png', 'cinder_blocks_1.png', 'wood_2.png',
                               'ceiling_tiles_1.png',   # to avoid aliasing
                               'lg_style_03_wall_purple_d_result.png', 'airduct_grate_1.png',
                               'lg_style_03_wall_orange_1.png', 'grass_2.png',
                               'lg_style_01_wall_light_m_result.png',
                               'lg_style_04_wall_purple_d_result.png',
                               'lg_style_03_floor_light1_m_result.png',
                               'lg_style_05_wall_red_d_result.png', 'slime_1.png',
                               'lg_style_05_wall_yellow_d_result.png', 'floor_tiles_bw_1.png',
                               'lg_style_02_floor_orange_d_result.png', 'lg_style_05_wall_yellow_bright_d_result.png',
                               'concrete_1.png', 'lg_style_03_wall_gray_d_result.png',
                               'lg_style_04_wall_red_d_result.png',   # to avoid aliasing
                               'lg_style_04_floor_orange_bright_d_result.png',
                               'lg_style_01_floor_orange_bright_d_result.png', 'stucco_1.png',
                               'lg_style_04_wall_green_bright_d_result.png', 'door_steel_brown.png',
                               'lg_style_03_floor_blue_bright_d_result.png', 'lava_1.png',
                               'lg_style_05_floor_light1_m_result.png',
                               'lg_style_01_wall_red_bright_1.png',
                               'lg_style_01_wall_green_1.png', 'lg_style_01_wall_yellow_1.png',
                               'lg_style_01_wall_red_1.png', 'lg_style_02_wall_yellow_d_result.png', 'door_doom_1.png',
                               'wood_planks_1.png', 'lg_style_03_floor_blue_d_result.png',
                               'lg_style_04_floor_blue_d_result.png', 'lg_style_03_floor_orange_d_result.png',
                               'lg_style_04_wall_red_bright_d_result.png', 'lg_style_02_floor_blue_bright_d_result.png',
                               'door_garage_white.png', 'lg_style_04_floor_blue_bright_d_result.png',
                               'lg_style_01_floor_blue_d_result.png',
                               'lg_style_02_floor_light_m_result.png',
                               'marble_2.png', 'lg_style_04_floor_cyan_d_result.png',
                               'lg_style_05_floor_blue_d_result.png', 'lg_style_01_wall_cerise_1.png',
                               'lg_style_02_wall_yellow_bright_d_result.png',
                               'lg_style_01_floor_blue_bright_d_result.png', 'lg_style_04_wall_green_d_result.png',
                               'drywall_1.png', 'lg_style_01_floor_blue_team_d_result.png', 'door_steel_red.png',
                               'lg_style_01_floor_light_m_result.png',
                               'lg_style_03_wall_cyan_1.png', 'marble_1.png',
                               'picket_fence_1.png', 'door_steel_grey.png', 'water_1.png',
                               'lg_style_02_floor_green_d_result.png', 'lg_style_01_floor_orange_d_result.png',
                               'lg_style_01_wall_green_bright_1.png', 'lg_style_03_floor_green_bright_d_result.png',
                               'lg_style_04_floor_orange_d_result.png', 'door_garage_red.png', 'brick_wall_1.png',
                               'lg_style_03_wall_gray_bright_d_result.png', 'lg_style_03_wall_blue_d_result.png',
                               'rock_1.png', 'lg_style_05_wall_red_bright_d_result.png', 'grass_1.png',
                               'lg_style_03_floor_green_d_result.png', 'lg_style_02_floor_green_bright_d_result.png',
                               'lg_style_05_floor_orange_d_result.png', 'door_doom_2.png',
                               'lg_style_02_wall_blue_d_result.png', 'lg_style_04_floor_dorange_d_result.png',
                               'lg_style_03_floor_purple_d_result.png', 'lg_style_05_floor_orange_bright_d_result.png',
                               'lg_style_01_floor_red_team_d_result.png', 'metal_grill_1.png',
                               'lg_style_02_floor_blue_d_result.png', 'cardboard_3.png',
                               'lg_style_01_ceiling_d_result.png', 'lg_style_01_wall_purple_1.png',
                               'lg_style_03_wall_orange_bright_d_result.png',
                               'lg_style_02_wall_blue_bright_d_result.png', 'cardboard_1.png',
                               'ceiling_tile_noborder_1.png', 'lg_style_02_wall_lgreen_d_result.png',
                               'lg_style_03_floor_red_d_result.png']

    return HARDCODED_TEXTURE_FILES


def draw_layout_overview(rooms_per_side, render_scale, texture_dir, add_textures=True):
    textures = load_textures(texture_dir)
    layout = define_layout(rooms_per_side, texture_dir)

    # draw texture background
    n_textures = len(textures)
    res = int(layout.table_size * render_scale)
    room_size = int(res / rooms_per_side)
    img = np.ones((res, res, 3))
    if add_textures:
        for x in range(rooms_per_side):
            for y in range(rooms_per_side):
                texture = cv2.imread(os.path.join(texture_dir,
                                                  textures[(x * rooms_per_side + y) % n_textures]))
                texture = cv2.resize(texture, (room_size, room_size))[:, :, ::-1] / 255.
                img[int(y * room_size) : int((y+1) * room_size),
                    int(x * room_size) : int((x+1) * room_size)] = texture

    def _add_horizontal_line(x_range, y):
        cv2.line(img, (int(x_range[0] * render_scale), res - int(y * render_scale - 1)),
                      (int(x_range[1] * render_scale), res - int(y * render_scale - 1)), (0, 0, 0), 3)
        return [[None]]

    def _add_vertical_line(y_range, x):
        cv2.line(img, (int(x * render_scale), res - int(y_range[0] * render_scale - 1)),
                      (int(x * render_scale), res - int(y_range[1] * render_scale - 1)), (0, 0, 0), 3)
        return [[None]]

    define_layout_raw(rooms_per_side, _add_horizontal_line, _add_vertical_line)
    return img


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    img = draw_layout_overview(rooms_per_side=5, render_scale=256, texture_dir="../../../../../assets/textures")
    # plt.imshow(img)
    # plt.show()
    plt.imsave("test.png", img)

    # import matplotlib.pyplot as plt
    # l = define_layout(10)
    # print(l.doors)
    # plt.scatter(l.ox, l.oy, c='black')
    # plt.axis('equal')
    # plt.show()
