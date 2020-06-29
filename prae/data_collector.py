import os
from itertools import count
import h5py


class DataCollector:
    """
    """

    def __init__(self, env, policy, data_folder):
        """
        """
        self.env = env
        self.policy = policy
        self.data_folder = os.path.join("data", data_folder)

    def collect(self, seed, n_episodes=100, dset="train"):
        """
        """
        replay_buffer = []

        start_ep = get_start_episode(self.data_folder, seed, dset)
        print(f"Sampling {dset} trajectories {start_ep} to "
              f"{start_ep+n_episodes}")

        for i_episode in range(start_ep, start_ep+n_episodes):
            trajectory = {"action": [],
                          "reward": [],
                          "states": [],
                          "done": []}
            episode_path = goc_episode_path(self.data_folder, seed, dset,
                                            i_episode)

            state = self.env.reset()

            trajectory["states"].append(state)

            for t in count():

                action = self.policy.select_action(state)

                state, reward, done, _ = self.env.step(action)

                trajectory["action"].append(action)
                trajectory["reward"].append(reward)
                trajectory["done"].append(int(done))
                trajectory["states"].append(state)

                if done:
                    save_dict_h5py(trajectory, os.path.join(episode_path,
                                                            "trajectory.h5py"))
                    break


def goc_episode_path(data_folder, seed, dset, i_ep):
    """
    Get_or_create function for episode path
    """
    episode_path = os.path.join(data_folder, str(seed), dset, f"{i_ep}")
    if not os.path.exists(episode_path):
        os.makedirs(episode_path)
    return episode_path


def get_start_episode(data_folder, seed, dset):
    """
    Get last sampled episode
    """
    data_path = os.path.join(data_folder, str(seed), dset)
    if not os.path.exists(data_path):
        return 0
    else:
        folders = sorted([int(x) for x in os.listdir(data_path)])
        return int(folders[-1]) + 1


def save_dict_h5py(data, fname):
    """
    Save dictionary containing numpy arrays to h5py file.
    """
    with h5py.File(fname, 'w') as hf:
        for key in data.keys():
            hf.create_dataset(key, data=data[key])
