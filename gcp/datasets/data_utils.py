import numpy as np


class DataSubsampler:
    def __init__(self, aggregator):
        self._aggregator = aggregator

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("This function needs to be implemented by sub-classes!")


class FixedFreqSubsampler(DataSubsampler):
    """Subsamples input array's first dimension by skipping given number of frames."""
    def __init__(self, n_skip, aggregator=None):
        super().__init__(aggregator)
        self._n_skip = n_skip

    def __call__(self, val, idxs=None, aggregate=False):
        """Subsamples with idxs if given, aggregates with aggregator if aggregate=True."""
        if self._n_skip == 0:
            return val, None

        if idxs is None:
            seq_len = val.shape[0]
            idxs = np.arange(0, seq_len - 1, self._n_skip + 1)

        if aggregate:
            assert self._aggregator is not None     # no aggregator given!
            return self._aggregator(val, idxs), idxs
        else:
            return val[idxs], idxs


class TargetLengthSubsampler(DataSubsampler):
    """Subsamples input array's first dimension equidistantly to the given target length."""

    def __init__(self, target_len, aggregator=None):
        super().__init__(aggregator)
        self._target_len = target_len

    def __call__(self, val, seq_len, idxs=None, aggregate=False):
        """Subsamples with idxs if given, aggregates with aggregator if aggregate=True."""
        if isinstance(val, int) or len(val.shape) == 0:
            return val, None
        elif seq_len <= self._target_len:
            return val[:self._target_len], None

        if idxs is None:
            sl = seq_len - 1      # -1 because action sequence is one shorter
            idxs = np.linspace(0, sl - 1, self._target_len, dtype=int)

        if aggregate:
            assert self._aggregator is not None  # no aggregator given!
            return self._aggregator(val, idxs), idxs
        else:
            return val[idxs], idxs

class Aggregator:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("This function needs to be implemented by sub-classes!")


class SumAggregator(Aggregator):
    def __call__(self, val, idxs):
        return np.add.reduceat(val, idxs, axis=0)


class RoomBasedSubsampler(DataSubsampler):
    """Subsamples input array's first dimension by skipping given number of frames."""

    SCALE = 27.0        # scale between mujoco and miniworld environment, also hardcoded in miniworld package

    def __init__(self, n_skip, n_rooms, subsample_rooms, aggregator=None):
        super().__init__(aggregator)
        self._n_skip = n_skip
        self._subsampler = FixedFreqSubsampler(n_skip, aggregator)
        self._get_current_room = self._get_room_fcn(n_rooms)
        self._subsample_rooms = subsample_rooms

    def __call__(self, val, states, idxs=None, aggregate=False):
        """Subsamples inside specified rooms (or at indx if given), aggregates with aggregator if aggregate=True."""
        if self._n_skip == 0 or idxs is not None:
            return self._subsampler(val, idxs, aggregate)

        assert states.shape[-1] >= 2  # need coordinate state
        rooms = [self._get_current_room(s[:2]) for s in states]
        subsampled_vals, subsample_idxs = np.array([]), np.array([])
        idx, current_room, last_room_change_idx = 0, rooms[0], 0
        while idx < len(rooms):
            if rooms[idx] != current_room:
                # room change detected -> subsample last room's subsequence if necessary
                subseq = val[last_room_change_idx:idx]
                idxs = np.asarray(list(range(last_room_change_idx, idx)))
                if current_room in self._subsample_rooms:
                    subseq, local_sample_idxs = self._subsampler(subseq, aggregate=aggregate)
                    idxs = idxs[local_sample_idxs]

                # append subseqs
                if subsampled_vals.size == 0:
                    subsampled_vals, subsample_idxs = subseq, idxs
                else:
                    subsampled_vals = np.concatenate((subsampled_vals, subseq), axis=0)
                    subsample_idxs = np.concatenate((subsample_idxs, idxs))
                current_room = rooms[idx]
                last_room_change_idx = idx

            idx += 1
        return subsampled_vals, subsample_idxs

    def _get_room_fcn(self, n_rooms):
        from gcp.infra.envs.miniworld_env.utils.multiroom2d_layout import define_layout
        raw_fcn = define_layout(int(np.sqrt(n_rooms))).coords2ridx

        def fcn(coords):
            c = coords / self.SCALE
            return raw_fcn(c[0], c[1])

        return fcn



