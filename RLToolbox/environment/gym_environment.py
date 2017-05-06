"""
`SpaceConversionEnv` acts as a wrapper on
any environment. It allows to convert some action spaces, and observation spaces to others.
"""

import numpy as np
from gym.spaces import Discrete, Box, Tuple
from gym import Env
import cv2
import gym

class Environment(Env):

    def __init__(self, env, pms, type="origin"):
        self.env = env
        self.type = type
        self.pms = pms

    def step(self, action, **kwargs):
        self._observation, reward, done, info = self.env.step(action)
        self._observation = np.clip(self._observation, self.env.observation_space.low, self.env.observation_space.high)
        return self.observation, reward, done, info

    def reset(self, **kwargs):
        self._observation = self.env.reset()
        return self.observation

    def render(self, mode="human", close=False):
        if mode == "human":
            return self.env.render(mode)
        elif mode == "rgb_array":
            return cv2.resize(self.env.render('rgb_array'), (self.pms.obs_shape[1], self.pms.obs_shape[0]))

    @property
    def observation(self):
        return self._observation
