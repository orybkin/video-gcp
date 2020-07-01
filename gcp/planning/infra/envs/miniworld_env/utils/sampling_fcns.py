import numpy as np


def _rand(n):
    return np.random.rand(n) - 0.5      # sample in range [-0.5, 0.5]


def _M(*elems):
    return np.array(elems)


class RoomSampler2d:
    """Samples 2d coordinates in a room environment with given number of rooms."""
    def __init__(self, rooms_per_side, sample_wide=False):
        """If sample_wide is True, the sampling fills the whole room all the way to the walls."""
        super().__init__()
        self._rooms_per_side = rooms_per_side

        self._agent_size = 0.02
        self._sampling_width = 1/3 - (not sample_wide) * 3 * self._agent_size      # equivalent to width of one room in mujoco
        self._room_offset = 1/3       # equivalent to middle point of rooms
        self._door_sampling_width = 1.5 * 0.0667 - 3 * self._agent_size

        self._hor_door_sampling_width = _M(2*self._agent_size, self._door_sampling_width)
        self._vert_door_sampling_width = _M(self._door_sampling_width, 2 * self._agent_size)

    def sample(self, room=None):
        if room is None:
            room = np.random.randint(self._rooms_per_side**2)
        room = self._ridx2coords(room)
        center = _M(*[self._room_offset / 2 + i * self._room_offset
                     - self._rooms_per_side / 2 * self._room_offset for i in room])  # centered around (0, 0)
        return _rand(2) * self._sampling_width + center

    def sample_door(self, room1, room2, sample_center=False):
        """Samples in the door way between two given rooms."""
        center = self.get_door_pos(room1, room2)
        if sample_center: return center
        room1, room2 = self._ridx2coords(room1), self._ridx2coords(room2)
        if room1[0] != room2[0] and room1[1] == room2[1]:
            # horizontal room connection
            return _rand(2) * self._hor_door_sampling_width + center
        elif room1[0] == room2[0] and room1[1] != room2[1]:
            # vertical room connection
            return _rand(2) * self._vert_door_sampling_width + center
        else:
            raise ValueError("Rooms don't have connection for door.")

    def get_door_pos(self, room1, room2):
        assert room1 < room2  # room1 needs to be on top or left of room2
        room1, room2 = self._ridx2coords(room1), self._ridx2coords(room2)
        assert np.abs(
            room1[0] - room2[0] + room1[1] - room2[1]) == 1  # difference between given rooms needs to be exactly 1
        center = _M(*[self._room_offset / 2 + (i + j) / 2 * self._room_offset
                      - self._rooms_per_side / 2 * self._room_offset for i, j in zip(room1, room2)])
        return center

    def get_door_path(self, room1, room2):
        """Returns path through the door between two rooms (i.e. three waypoints)."""
        lefttop = room1 < room2     # check if room 1 is on left/top of room2
        center = self.get_door_pos(min(room1, room2), max(room1, room2))
        room1, room2 = self._ridx2coords(room1), self._ridx2coords(room2)
        if room1[0] != room2[0] and room1[1] == room2[1]:
            # horizontal room connection
            offset = _M(3 * self._door_sampling_width, 0)
        elif room1[0] == room2[0] and room1[1] != room2[1]:
            # vertical room connection
            offset = _M(0, -3 * self._door_sampling_width)
        else:
            raise ValueError("Rooms don't have connection for door.")
        if lefttop:
            return [center - offset, center, center + offset]
        else:
            return [center + offset, center, center - offset]

    def _ridx2coords(self, room_idx):
        """Converts room index to coordinates based on grid size."""
        return (int(np.floor(room_idx / self._rooms_per_side)),
                int(self._rooms_per_side - 1 - room_idx % self._rooms_per_side))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    DOORS = [(0, 3), (3, 6), (1, 4), (4, 7), (5, 8), (3, 4), (1, 2), (7, 8)]
    sampler = RoomSampler2d(rooms_per_side=3, sample_wide=True)
    samples = np.asarray([[sampler.sample(r) for _ in range(100)] for r in range(36)]).transpose(2, 0, 1).reshape(2, -1)
    plt.scatter(samples[0], samples[1], c='black')

    samples = np.asarray([[sampler.sample_door(d[0], d[1]) for _ in range(10)] for d in DOORS]).transpose(2, 0, 1).reshape(2, -1)
    plt.scatter(samples[0], samples[1], c='red')
    plt.show()

