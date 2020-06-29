import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import gym
from gym import spaces
from gym.utils import seeding

import itertools


class FashionMnistTranslate2dEnv(gym.Env):
    """
    """
    def __init__(self, width=32, height=32, seed=None,
                 train=True, random_goal=True, goal_mode="train", data=None):
        """
        """
        self.width = width
        self.height = height

        self.max_offset = 3
        self.max_time = 100
        self.max_offset = 3

        self.np_random = None

        self.time = 0

        self.num_actions = 4
        self.action_space = spaces.Discrete(self.num_actions)

        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(self.width, self.height),
            dtype=np.float32
        )

        self.train = train
        self.goal_mode = goal_mode

        self.data = data

        self.seed(seed)

    def __str__(self):
        """
        """
        name = self.__class__.__name__
        return f"Env: {name}, train: {self.train}, goal mode: {self.goal_mode}"

    def seed(self, seed=None):
        """
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render_image(self, game, trans):
        """
        """
        pad = (self.width - game.size[0], self.height - game.size[1])

        image = Image.new('RGB', (game.size[0]+pad[0], game.size[1]+pad[1]))

        image.paste(
            game.rotate(0, resample=Image.BICUBIC),
            (pad[0] // 2 + trans[0],
             pad[1] // 2 + trans[1]))
        return np.asarray(image.convert('L'))[None, :, :]/255.

    def render(self):
        """
        Render the current image translation
        """
        im = self.render_image(self.game, self.trans)
        return im

    def create_goal(self, digit):
        """
        """
        if self.goal_mode == 'train':
            # At train time, all the goals are in [-3, y] up to and including
            # [-1, y]
            x_range = np.arange(-self.max_offset, 0)
            y_range = np.arange(-self.max_offset, self.max_offset+1)

        elif self.goal_mode == 'test':
            # At test time, all the goals are in [1, y], up to and including
            # [3, y]
            x_range = np.arange(1, self.max_offset+1)
            y_range = np.arange(-self.max_offset, self.max_offset+1)
        self.goal_trans = [np.random.choice(x_range),
                           np.random.choice(y_range)]
        return self.render_image(self.game, self.goal_trans), self.goal_trans

    def get_digit(self):
        """
        """
        max_range = len(self.data)
        idx = np.random.randint(max_range)
        return idx

    def reset(self):
        """
        """
        self.time = 0
        self.current_digit = self.get_digit()

        self.game = self.data[self.current_digit][0]

        self.trans = self.random_translation()

        self.goal, self.goal_trans = self.create_goal(self.game)

        return self.render()

    def random_translation(self):
        """
        Randomly sample an x and y translation
        """
        x = np.arange(-self.max_offset, self.max_offset + 1)
        y = np.arange(-self.max_offset, self.max_offset + 1)
        trans = [np.random.choice(x), np.random.choice(y)]
        return trans

    def get_goal_state(self):
        """
        """
        trans = self.goal_trans
        digit = self.current_digit
        game = self.data[digit][0]
        goal_state = self.render_image(game, trans)
        return goal_state

    def translate(self, translation=(0, 0)):
        """
        """
        self.trans[0] = int(
            np.clip(self.trans[0] + translation[0], -self.max_offset,
                    self.max_offset))
        self.trans[1] = int(
            np.clip(self.trans[1] + translation[1], -self.max_offset,
                    self.max_offset))

    def at_goal(self):
        """
        """
        if self.goal_trans[0] == self.trans[0] and self.goal_trans[1] == self.trans[1]:
            r = 1.
        else:
            r = 0.
        return r


    def step(self, action):
        """
        """
        done = False
        reward = 0.
        self.time += 1

        if action == 0:
            # Move 1 px north.
            self.translate((-1, 0))
        elif action == 1:
            # Move 1 px east.
            self.translate((0, 1))
        elif action == 2:
            # Move 1 px south.
            self.translate((1, 0))
        elif action == 3:
            # Move 1 px west.
            self.translate((0, -1))
        else:
            raise ValueError("Invalid action.")

        reward = self.at_goal()
        if reward == 1.:
            done = True

        if self.time == self.max_time:
            done = True

        return self.render(), reward, done, None
