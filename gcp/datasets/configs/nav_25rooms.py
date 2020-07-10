from blox import AttrDict
from gcp.datasets.configs.nav_9rooms import Nav9Rooms


class Nav25Rooms(Nav9Rooms):
    n_rooms = 25


config = AttrDict(
    dataset_spec=AttrDict(
        max_seq_len=200,
        dataset_class=Nav25Rooms,
        split=AttrDict(train=0.994, val=0.006, test=0.00),
    ),
    n_rooms=25,
    crop_window=40,
)

