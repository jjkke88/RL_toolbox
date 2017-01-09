import numpy as np


class Baseline(object):
    def fit(self, paths):
        self.temp = 0

    def predict(self, path):
        return np.zeros(len(path["rewards"]))