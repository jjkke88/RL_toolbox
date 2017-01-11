from parameters import PMS_dqn
import os
import tensorflow as tf
import tensorlayer as tl
from RLToolbox.agent.DQN_agent import DQNAgent
from RLToolbox.environment.gym_environment import Environment
import gym
from RLToolbox.storage.storage_replay import StorageReplay
from RLToolbox.toolbox.common.utils import *
class DQNNet(object):
    def __init__(self, scope, pms):
        self.obs = tf.placeholder(dtype=tf.float32, shape=[None, pms.obs_shape], name="%s_obs"%scope)
        self.action_acted = tf.placeholder(dtype=tf.float32, shape=[None, pms.action_shape], name="%s_action"%scope)
        self.target_value = tf.placeholder(dtype=tf.float32, shape=[None], name='%s_target_value'%scope)
        with tf.variable_scope("%s_q" % scope):
            network = tl.layers.InputLayer(self.obs, name='%s_q_input_layer' % scope)
            network = tl.layers.DenseLayer(network, n_units=64,
                                           act=tf.nn.relu, name="%s_q_fc1" % scope)
            network = tl.layers.DenseLayer(network, n_units=64,
                                           act=tf.nn.relu, name="%s_q_fc2" % scope)
            network = tl.layers.DenseLayer(network, n_units=pms.action_shape,
                                           act=tf.nn.relu, name="%s_q_q" % scope)

            self.q = network.outputs
            self.action_n = tf.argmax(self.q, 1)
            self.q_n = tf.reduce_sum(self.q * self.action_acted, reduction_indices=1, name='q_acted')
            self.delta = self.target_value - self.q_n
            self.loss = tf.reduce_mean(0.5 * tf.square(self.delta), name='loss')
            self.learning_rate_step = tf.placeholder(dtype='int64', shape=None, name='%s_learning_rate_step'%scope)
            self.learning_rate_op = tf.maximum(pms.learning_rate_minimum,
                                               tf.train.exponential_decay(
                                                   pms.learning_rate,
                                                   self.learning_rate_step,
                                                   pms.learning_rate_decay_step,
                                                   pms.learning_rate_decay,
                                                   staircase=True))
            self.opt = tf.train.GradientDescentOptimizer(
                self.learning_rate_op).minimize(self.loss)

        with tf.variable_scope("%s_target" % scope):
            target_network = tl.layers.InputLayer(self.obs, name='%s_target_input_layer' % scope)
            target_network = tl.layers.DenseLayer(target_network, n_units=64,
                                           act=tf.nn.relu, name="%s_target_fc1" % scope)
            target_network = tl.layers.DenseLayer(target_network, n_units=64,
                                           act=tf.nn.relu, name="%s_target_fc2" % scope)
            target_network = tl.layers.DenseLayer(target_network, n_units=pms.action_shape,
                                           act=tf.nn.relu, name="%s_target_q" % scope)
            self.target_q = target_network.outputs

        self.var_list = [v for v in tf.trainable_variables() if v.name.startswith("%s_q" % scope)]
        self.var_list_target = [v for v in tf.trainable_variables() if v.name.startswith("%s_target" % scope)]
        self.gf = GetFlat(self.var_list)
        self.sff = SetFromFlat(self.var_list_target)



    def update_target_net(self):
        self.sff(self.gf())

if __name__ == "__main__":
    if not os.path.isdir("./checkpoint"):
        os.makedirs("./checkpoint")
    if not os.path.isdir("./log"):
        os.makedirs("./log")
    pms = PMS_dqn().pms
    env = Environment(gym.make(pms.environment_name), pms=pms)
    session = tf.Session()
    net = DQNNet("dqn", pms)
    agent = DQNAgent(env, session, None, None, None, net, pms)
    storage = StorageReplay(agent, env, None, pms)
    agent.storage = storage
    agent.init_network()
    agent.saver = tf.train.Saver(max_to_keep=10)
    if pms.train_flag:
        agent.learn()
    else:
        agent.test(pms.checkpoint_file)