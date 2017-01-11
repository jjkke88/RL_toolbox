import numpy as np
class Baseline(object):
    def __init__(self, reg_coeff=1e-5):
        self._coeffs = None
        self._reg_coeff = reg_coeff

    def get_param_values(self, **tags):
        return self._coeffs

    def set_param_values(self, val, **tags):
        self._coeffs = val

    def _features(self, path):
        o = path["observations"].astype('float32')
        o = o.reshape(o.shape[0], -1)
        l = len(path["rewards"])
        al = np.arange(l).reshape(-1 , 1) / 100.0
        return np.concatenate([o, o ** 2, al, al ** 2, np.ones((l, 1))], axis=1)

    def fit(self, paths):
        featmat = np.concatenate([self._features(path) for path in paths])
        returns = np.concatenate([path["returns"] for path in paths])
        self._coeffs = np.linalg.lstsq(
            featmat.T.dot(featmat) + self._reg_coeff * np.identity(featmat.shape[1]),
            featmat.T.dot(returns)
        )[0]

    def predict(self, path):
        if self._coeffs is None:
            return np.zeros(len(path["rewards"]))
        return self._features(path).dot(self._coeffs)
