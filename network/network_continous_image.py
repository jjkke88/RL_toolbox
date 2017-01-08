from utils import *
import numpy as np
import tensorflow as tf
import prettytensor as pt
from parameters import pms

seed = 1
np.random.seed(seed)
tf.set_random_seed(seed)

class NetworkContinousImage(object):
    def __init__(self, scope):
        with tf.variable_scope("%s_shared" % scope):
            self.obs = obs = tf.placeholder(
                dtype, shape=[None, pms.obs_height, pms.obs_width, pms.obs_channel], name="%s_obs"%scope)
            self.action_n = tf.placeholder(dtype, shape=[None, pms.action_shape], name="%s_action"%scope)
            self.advant = tf.placeholder(dtype, shape=[None], name="%s_advant"%scope)
            self.old_dist_means_n = tf.placeholder(dtype, shape=[None, pms.action_shape],
                                                   name="%s_oldaction_dist_means"%scope)
            self.old_dist_logstds_n = tf.placeholder(dtype, shape=[None, pms.action_shape],
                                                     name="%s_oldaction_dist_logstds"%scope)
            self.action_dist_means_n = (pt.wrap(self.obs).
                                        conv2d(8 , 32 , stride=4 , batch_normalize=True).
                                        conv2d(4 , 64 , stride=2 , batch_normalize=True).
                                        conv2d(3 , 64 , stride=1 , batch_normalize=True).
                                        flatten().
                                        fully_connected(128, activation_fn=tf.nn.relu, init=tf.random_normal_initializer(-0.05, 0.05),
                                                        name="%s_fc1"%scope).
                                        fully_connected(128, activation_fn=tf.nn.relu, init=tf.random_normal_initializer(-0.05, 0.05),
                                                         name="%s_fc2"%scope).
                                        fully_connected(pms.action_shape, init=tf.random_normal_initializer(-0.05, 0.05),
                                                        name="%s_fc3"%scope))

            self.N = tf.shape(obs)[0]
            Nf = tf.cast(self.N, dtype)
            self.action_dist_logstd_param = tf.Variable((.01*np.random.randn(1, pms.action_shape)).astype(np.float32),  name="%spolicy_logstd"%scope)
            self.action_dist_logstds_n = tf.tile(self.action_dist_logstd_param,
                                              tf.pack((tf.shape(self.action_dist_means_n)[0], 1)))
            self.var_list = [v for v in tf.trainable_variables()if v.name.startswith(scope)]

    def get_action_dist_means_n(self, session, obs):
        return session.run(self.action_dist_means_n,
                         {self.obs: obs})

