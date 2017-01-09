import os

import gym
import numpy as np
import prettytensor as pt
import tensorflow as tf
import tensorlayer as tl
from RLToolbox.agent.TRPO_agent import TRPOAgent
from RLToolbox.environment.gym_environment import Environment
from RLToolbox.storage.storage_continous import Storage
from RLToolbox.toolbox.baseline.baseline_lstsq import Baseline
from RLToolbox.toolbox.distribution.diagonal_gaussian import DiagonalGaussian

from RLToolbox.parameters import PMS_base


class NetworkContinous(object):
    def __init__(self, scope):
        with tf.variable_scope("%s_shared" % scope):
            self.obs = obs = tf.placeholder(
                tf.float32, shape=[None, pms.obs_shape], name="%s_obs"%scope)
            self.action_n = tf.placeholder(tf.float32, shape=[None, pms.action_shape], name="%s_action"%scope)
            self.advant = tf.placeholder(tf.float32, shape=[None], name="%s_advant"%scope)
            self.old_dist_means_n = tf.placeholder(tf.float32, shape=[None, pms.action_shape],
                                                   name="%s_oldaction_dist_means"%scope)
            self.old_dist_logstds_n = tf.placeholder(tf.float32, shape=[None, pms.action_shape],
                                                     name="%s_oldaction_dist_logstds"%scope)
            self.action_dist_means_n = (pt.wrap(self.obs).
                                        fully_connected(64, activation_fn=tf.nn.relu, init=tf.random_normal_initializer(-0.05, 0.05), bias_init=tf.constant_initializer(0),
                                                        name="%s_fc1"%scope).
                                        fully_connected(64, activation_fn=tf.nn.relu, init=tf.random_normal_initializer(-0.05, 0.05), bias_init=tf.constant_initializer(0),
                                                         name="%s_fc2"%scope).
                                        fully_connected(pms.action_shape, init=tf.random_normal_initializer(-0.05, 0.05), bias_init=tf.constant_initializer(0),
                                                        name="%s_fc3"%scope))

            self.N = tf.shape(obs)[0]
            Nf = tf.cast(self.N, tf.float32)
            self.action_dist_logstd_param = tf.Variable((.01*np.random.randn(1, pms.action_shape)).astype(np.float32), name="%spolicy_logstd"%scope)
            self.action_dist_logstds_n = tf.tile(self.action_dist_logstd_param,
                                              tf.pack((tf.shape(self.action_dist_means_n)[0], 1)))
            self.var_list = [v for v in tf.trainable_variables()if v.name.startswith(scope)]
class NetworkTL(object):
    def __init__(self, scope):
        with tf.variable_scope("%s_shared" % scope):
            self.obs = obs = tf.placeholder(
                tf.float32 , shape=[None , pms.obs_shape] , name="%s_obs" % scope)
            self.action_n = tf.placeholder(tf.float32 , shape=[None , pms.action_shape] , name="%s_action" % scope)
            self.advant = tf.placeholder(tf.float32 , shape=[None] , name="%s_advant" % scope)
            self.old_dist_means_n = tf.placeholder(tf.float32 , shape=[None , pms.action_shape] ,
                                                   name="%s_oldaction_dist_means" % scope)
            self.old_dist_logstds_n = tf.placeholder(tf.float32 , shape=[None , pms.action_shape] ,
                                                     name="%s_oldaction_dist_logstds" % scope)
            network = tl.layers.InputLayer(self.obs , name='%s_input_layer'%scope)
            network = tl.layers.DenseLayer(network , n_units=64 ,
                                           act=tf.nn.relu , name="%s_fc1"%scope)
            network = tl.layers.DenseLayer(network , n_units=64 ,
                                           act=tf.nn.relu , name="%s_fc2"%scope)
            network = tl.layers.DenseLayer(network , n_units=pms.action_shape ,
                                           act=tf.nn.relu , name="%s_fc3"%scope)
            self.action_dist_means_n = network.outputs
            self.N = tf.shape(obs)[0]
            Nf = tf.cast(self.N , tf.float32)
            self.action_dist_logstd_param = tf.Variable(
                (.01 * np.random.randn(1 , pms.action_shape)).astype(np.float32) , name="%spolicy_logstd" % scope)
            self.action_dist_logstds_n = tf.tile(self.action_dist_logstd_param ,
                                                 tf.pack((tf.shape(self.action_dist_means_n)[0] , 1)))
            self.var_list = [v for v in tf.trainable_variables() if v.name.startswith(scope)]

if __name__ == "__main__":
    if not os.path.isdir("./checkpoint"):
        os.makedirs("./checkpoint")
    if not os.path.isdir("./log"):
        os.makedirs("./log")
    pms = PMS_base().pms
    # pms.train_flag = False
    # pms.render = True
    env = Environment(gym.make(pms.environment_name), pms=pms)
    session = tf.Session()
    net = NetworkTL("continous")
    baseline = Baseline()
    storage = None
    distribution = DiagonalGaussian(pms.action_shape)
    agent = TRPOAgent(env, session, baseline, storage, distribution, net, pms)
    agent.storage = Storage(agent , env , baseline, pms)
    if pms.train_flag:
        agent.learn()
    else:
        agent.test(pms.checkpoint_file)
    # env.monitor.close()
    # gym.upload(training_dir,
    #            algorithm_id='trpo_ff')
