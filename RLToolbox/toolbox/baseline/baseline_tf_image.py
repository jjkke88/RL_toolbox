import numpy as np
import prettytensor as pt
import tensorflow as tf


class BaselineTfImage(object):
    coeffs = None

    def __init__(self, session):
        self.net = None
        self.session = session

    def create_net(self, shape):
        self.x = tf.placeholder(tf.float32, shape=[None, shape[1], shape[2], shape[3]], name="x")
        self.y = tf.placeholder(tf.float32, shape=[None], name="y")
        self.net = (pt.wrap(self.x).
                    conv2d(1, 16, stride=2, batch_normalize=True).
                    conv2d(1, 16, stride=2, batch_normalize=True).
                    flatten().
                    fully_connected(32, activation_fn=tf.nn.relu).
                    fully_connected(32, activation_fn=tf.nn.relu).
                    fully_connected(1))
        self.net = tf.reshape(self.net, (-1, ))
        l2 = (self.net - self.y) * (self.net - self.y)
        self.train = tf.train.AdamOptimizer().minimize(l2)
        self.session.run(tf.initialize_all_variables())

    def _features(self, path):
        ret = path["observations"].astype('float32')
        return ret

    def fit(self, paths):
        featmat = np.concatenate([self._features(path) for path in paths])
        if self.net is None:
            self.create_net(featmat.shape)
        returns = np.concatenate([path["returns"] for path in paths])
        for _ in range(100):
            self.session.run(self.train, {self.x: featmat, self.y: returns})

    def predict(self, path):
        if self.net is None:
            return np.zeros(len(path["rewards"]))
        else:
            ret = self.session.run(self.net, {self.x: self._features(path)})
            return np.reshape(ret, (ret.shape[0], ))