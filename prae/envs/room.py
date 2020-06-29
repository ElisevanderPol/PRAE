import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

import copy

import argparse


class RoomEnv(gym.Env):
    """
    """
    def __init__(self, width=8, height=8, seed=None,
                 train=True, random_goal=False, n_objects=1):
        """
        """
        self.width = width
        self.height = height

        self.max_time = 100

        if not n_objects in [1, 2]:
            raise ValueError(f"No implementation for {n_objects} objects."
                             "Provide either 1 or 2")
        self.n_objects = n_objects

        self.train = train
        self.random_goal = random_goal

        self.num_actions = 4

        self.action_space = spaces.Discrete(self.num_actions)

        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(self.width, self.height),
            dtype=np.float32
        )

        self.seed(seed)

        self.all_locs = self.get_locs()

        self.ep_count = 0.
        self.reset()

    def get_locs(self):
        """
        """
        all_locs = []
        for x in range(1, self.height-1):
            for y in range(1, self.width-1):
                all_locs.append([x, y])
        return all_locs

    def __str__(self):
        """
        """
        name = self.__class__.__name__
        return (f"Env: {name}, train: {self.train}, n_obj: {self.n_objects}, "
                f"random goal: {self.random_goal}")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_random_location(self):
        """
        """
        idx = np.random.randint(len(self.all_locs))
        loc = self.all_locs.pop(idx)
        return loc

    def get_locations(self):
        """
        """
        self.key = [5, 5]
        self.all_locs.remove(self.key)
        if self.n_objects > 1:
            self.mail = [3, 3]
            self.all_locs.remove(self.mail)
        else:
            self.mail = None
        self.robot = self.get_random_location()

    def reset(self):
        """
        """
        self.get_locations()
        if self.train:
            deliveries = [[1, 1], [6, 6], [1, 6], [6, 1]]
        elif not self.train:
            deliveries = self.all_locs
        deliv = np.random.randint(len(deliveries))
        self.deliveries = [deliveries[deliv]]

        self.ep_count += 1

        self.goal = self.get_goal()
        self.store_goal = copy.deepcopy(self.goal)
        # print("goal", self.goal, "train", self.train)

        self.current = ["key"]
        if self.n_objects > 1:
            self.current.append("mail")
        self.all_locs = self.get_locs()

        self.time = 0

        room = self.render_im(self.robot, self.key, self.mail)
        return room

    def render(self):
        """
        """
        room = self.render_im(self.robot, self.key, self.mail)
        return room

    def render_im(self, robot_loc, key_loc, mail_loc):
        """
        """
        key = self.render_key()
        mail = self.render_mail()
        robot = self.render_robot()
        room = np.zeros((self.height*6, self.width*6, 3))

        n = 6
        room[0:n, :, :] = 1
        room[:, 0:n, :] = 1
        room[n*self.height-n:n*self.height, :, :] = 1
        room[n*self.width-n:n*self.width, :, :] = 1
        room[:, n*self.height-n:n*self.height, :] = 1
        room[:, n*self.width-n-n*self.width, :] = 1


        robot_x, robot_y = robot_loc
        room[robot_x*n:robot_x*n+n, robot_y*n:robot_y*n+n, 0] = robot

        if key_loc is not None:
            key_x, key_y = key_loc
            room[key_x*n:key_x*n+n, key_y*n:key_y*n+n, 1] = key
        if mail_loc is not None:
            mail_x, mail_y = mail_loc
            room[mail_x*n:mail_x*n+n, mail_y*n:mail_y*n+n, 2] = mail
        return np.transpose(room, (2, 0, 1))

    def move_robot(self, vector):
        """
        """
        new_robot = [0, 0]
        new_robot[0] = int(
            np.clip(self.robot[0] + vector[0], 1,
                    self.width-2))
        new_robot[1] = int(
            np.clip(self.robot[1] + vector[1], 1,
                    self.height-2))
        # If the new state is in a wall:
        self.robot = new_robot
        reward = self.grab()
        return reward

    def grab(self):
        """
        """
        if self.robot == self.key:
            self.key = None
            self.current.remove("key")
            if "key" in self.goal:
                reward = 1.
                self.goal.remove("key")
            if "key" not in self.store_goal:
                reward = -1.
        elif self.robot == self.mail:
            self.mail = None
            self.current.remove("mail")
            if "mail" in self.goal:
                reward = 1.
                self.goal.remove("mail")
            if "mail" not in self.store_goal:
                reward = - 1.
        elif self.robot in self.deliveries and  len(self.goal) != len(self.store_goal):
                reward = 1.
        else:
            reward = -0.1
        return reward

    def step(self, action):
        """
        """
        self.time += 1.
        if action == 0:
            reward = self.move_robot((-1, 0))
        elif action == 1:
            reward = self.move_robot((0, 1))
        elif action == 2:
            reward = self.move_robot((1, 0))
        elif action == 3:
            reward = self.move_robot((0, -1))

        room = self.render_im(self.robot, self.key, self.mail)
        done = False

        if self.time == self.max_time:
            done = True
        # If we picked something up and are at delivery, episode ends
        if self.n_objects > len(self.current) and self.robot in self.deliveries:
            done = True
        return room, reward, done, ""

    def get_goal(self):
        """
        """
        # Random goal at train time
        if not self.random_goal:
            self.goal = ["key"]
        else:
            if self.n_objects == 1:
                goals = ["key"]
            elif self.n_objects == 2:
                goals = ["key", "mail"]
            g_idx = np.random.randint(len(goals))
            self.goal = [goals[g_idx]]
        return self.goal

    def get_goal_state(self):
        """
        """
        key_loc = self.key
        mail_loc = self.mail
        if "key" in self.goal:
            goal_room = self.render_im(self.deliveries[0], None, mail_loc)
        elif "mail" in self.goal:
            goal_room = self.render_im(self.deliveries[0], key_loc, None)
        return goal_room


    def render_key(self):
        """
        """
        key = np.zeros((6, 6))
        key[0][2] = 1.
        key[0][3] = 1.
        key[0][4] = 1.
        key[1][2] = 1.
        key[1][4] = 1.
        key[2][2] = 1.
        key[2][3] = 1.
        key[2][4] = 1.
        key[3][2] = 1.
        key[4][2] = 1.
        key[4][3] = 1.
        key[5][2] = 1.
        key[5][3] = 1.
        key[5][4] = 1.
        return key

    def render_mail(self):
        """
        """
        mail = np.zeros((6, 6))
        mail[1][0] = 1.
        mail[1][1] = 1.
        mail[1][2] = 1.
        mail[1][3] = 1.
        mail[1][4] = 1.
        mail[1][5] = 1.
        mail[2][0] = 1.
        mail[2][5] = 1.
        mail[3][0] = 1.
        mail[3][1] = 1.
        mail[3][2] = 1.
        mail[3][3] = 1.
        mail[3][4] = 1.
        mail[3][5] = 1.
        mail[4][0] = 1.
        mail[4][5] = 1.
        mail[5][0] = 1.
        mail[5][1] = 1.
        mail[5][2] = 1.
        mail[5][3] = 1.
        mail[5][4] = 1.
        mail[5][5] = 1.
        return mail

    def render_robot(self):
        """
        """
        robot = np.zeros((6, 6))
        robot[0][2] = 1.
        robot[1][0] = 1.
        robot[1][1] = 1.
        robot[1][2] = 1.
        robot[1][3] = 1.
        robot[1][4] = 1.
        robot[2][1] = 1.
        robot[2][2] = 1.
        robot[2][3] = 1.
        robot[3][1] = 1.
        robot[3][2] = 1.
        robot[3][3] = 1.
        robot[4][1] = 1.
        robot[4][3] = 1.
        robot[5][0] = 1.
        robot[5][1] = 1.
        robot[5][3] = 1.
        robot[5][4] = 1.
        return robot
