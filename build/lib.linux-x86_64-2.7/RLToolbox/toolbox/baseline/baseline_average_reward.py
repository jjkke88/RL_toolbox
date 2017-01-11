import numpy as np


class BaselineAverageReward(object):
    def fit(self, paths):
        self.temp = 0

    def predict(self, path):
        rewards = path["rewards"]
        mean_rewards = np.mean(rewards)
        return mean_rewards