import torch
import h5py
from torch.utils import data
import numpy as np
import os


class StateTransitionsDataset(data.Dataset):
    """
    Adaptation of https://github.com/tkipf/c-swm/blob/master/utils.py
    """
    def __init__(self, data_loc, hdf5_file=None, n_neg_samples=1,
                 num_actions=3):
        self.data = {}

        self.data_loc = data_loc

        self.idx2episode = list()
        step = 0

        for ep in sorted(os.listdir(data_loc)):
            traj = load_dict_h5py(os.path.join(data_loc, str(ep),
                                               "trajectory.h5py"))
            self.data[ep] = traj

            num_steps = len(traj['action'])
            idx_tuple = [(int(ep), idx) for idx in range(num_steps)]
            # Make dataset where each tuple is a data point
            self.idx2episode.extend(idx_tuple)
            step += num_steps
        self.experience_buffer = data
        self.num_steps = step
        self.data_dict = data

        print(f"Number of samples={self.num_steps}")

        self.n_neg_samples = n_neg_samples
        self.num_actions = num_actions

    def __len__(self):
        return self.num_steps

    def __getitem__(self, idx):
        ep, step = self.idx2episode[idx]
        traj = self.data[str(ep)]

        obs = to_float(traj["states"][step])
        action = traj["action"][step]
        reward = np.array([to_float(traj["reward"][step])])
        next_obs = to_float(traj["states"][step+1])

        num_steps = len(traj["action"])

        # Within episode negative sampling
        neg_idx = np.random.choice(np.arange(num_steps),
                                   size=self.n_neg_samples)
        if self.n_neg_samples > 0:
            neg_obs_list = [np.expand_dims(to_float(traj["states"][n]), axis=0)
                            for n in neg_idx]
        else:
            neg_obs_list = [np.expand_dims(to_float(np.zeros(obs.shape)),
                                           axis=0)]
        neg_samples = np.concatenate(neg_obs_list, axis=0)

        return obs, action, next_obs, reward, neg_samples


def to_float(np_array):
    """
    From https://github.com/tkipf/c-swm/blob/master/utils.py
    """
    return np.array(np_array, dtype=np.float32)


def load_dict_h5py(fname):
    """
    Restore dictionary containing numpy arrays from h5py file. (taken from
    Relemb code)
    From https://github.com/tkipf/c-swm/blob/master/utils.py
    """
    data = dict()
    with h5py.File(fname, 'r') as hf:
        for key in hf.keys():
            data[key] = hf[key][:]
    return data
