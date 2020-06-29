import numpy as np
import gym

class CartpoleWrapper:
    """
    """
    def __init__(self, seed):
        """
        """
        self.env = gym.make('CartPole-v0')
        self.env.seed(int(seed))
        self.action_space = self.env.action_space

    def step(self, action):
        """
        """
        s, r, d, i = self.env.step(action)
        return s, r, d, i

    def reset(self):
        """
        """
        s = self.env.reset()
        return s

    def get_goal_state(self):
        """
        """
        return np.array([[0., 0., 0., 0.]])
