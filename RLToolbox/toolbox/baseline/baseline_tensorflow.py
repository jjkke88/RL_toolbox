import tensorflow as tf
import numpy as np
import tensorlayer as tl

class Baseline(object):
    coeffs = None

    def __init__(self):
        self.net = None

    def create_net(self , shape):
        print(shape)
        self.g1 = tf.Graph()
        with self.g1.as_default() as g:
            self.x = tf.placeholder(tf.float32 , shape=[None, shape] , name="x")
            self.y = tf.placeholder(tf.float32 , shape=[None] , name="y")
            network = tl.layers.InputLayer(self.x , name='input_x')
            network = tl.layers.DenseLayer(network , n_units=128 , W_init=tf.truncated_normal_initializer(stddev=0.1) ,
                                           act=tf.nn.relu , name="dense_1")
            network = tl.layers.DenseLayer(network , n_units=128 , W_init=tf.truncated_normal_initializer(stddev=0.1) ,
                                           act=tf.nn.relu , name="dense_2")
            network = tl.layers.DenseLayer(network , n_units=1 ,
                                           name="v")
            self.net = network.outputs
            self.net = tf.reshape(self.net , (-1 ,))
            self.l2 = (self.net - self.y) * (self.net - self.y)
            self.train = tf.train.AdamOptimizer().minimize(self.l2)
        with self.g1.as_default():
            config = tf.ConfigProto(device_count={'GPU': 0})
            self.session = tf.Session(graph=self.g1, config=config)
            self.session.run(tf.global_variables_initializer())

    def _features(self, path):
        o = path["observations"].astype('float32')
        o = o.reshape(o.shape[0] , -1)
        l = len(path["rewards"])
        al = np.arange(l).reshape(-1 , 1) / 100.0
        return np.concatenate([o , o ** 2 , al , al ** 2 , np.ones((l , 1))] , axis=1)

    def fit(self, paths):
        featmat = np.concatenate([self._features(path) for path in paths])
        if self.net is None:
            self.create_net(featmat.shape[1])
        returns = np.concatenate([path["returns"] for path in paths])
        for _ in xrange(1):
            loss, _ = self.session.run([self.l2, self.train], {self.x: featmat , self.y: returns})

    def predict(self, path):
        if self.net is None:
            return np.zeros(len(path["rewards"]))
        else:
            return np.zeros(len(path["rewards"]))
            ret = self.session.run(self.net , {self.x: self._features(path)})
            return np.reshape(ret , (ret.shape[0] ,))
