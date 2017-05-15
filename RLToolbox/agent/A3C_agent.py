from RLToolbox.toolbox.common.utils import *
import numpy as np
import tensorflow as tf
import multiprocessing
import gym
from RLToolbox.toolbox.math import krylov
import time
import math
from RLToolbox.toolbox.logger.logger import Logger

seed = 1
np.random.seed(seed)
tf.set_random_seed(seed)

# tf.train.GradientDescentOptimizer().apply_gradients()
class A3CAgent(multiprocessing.Process):
    def __init__(self, env, session, baseline, storage, distribution, net, pms, task_q , result_q, process_id=0):
        multiprocessing.Process.__init__(self)
        self.process_id = process_id
        self.task_q = task_q
        self.result_q = result_q
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.pms = pms
        self.env_class = env
        self.baseline = baseline
        self.distribution = distribution
        self.action_net = net["action_net"]
        self.value_net = net["value_net"]

    def init_environment(self):
        self.env = self.env_class(gym.make(self.pms.environment_name), pms = self.pms)

    def get_gf_sff(self, net):
        # get getflag and setflag object
        var_list = net.var_list
        gf = GetFlat(var_list)  # get theta from var_list
        gf.session = self.session
        sff = SetFromFlat(var_list)  # set theta from var_List
        sff.session = self.session
        return gf, sff

    def init_network(self):
        config = tf.ConfigProto(device_count={'GPU': 0})
        config.gpu_options.per_process_gpu_memory_fraction = self.pms.GPU_fraction
        self.session = tf.Session(config=config)
        self.session.run(tf.global_variables_initializer())
        self.action_gf, self.action_sff = self.get_gf_sff(self.action_net)
        self.value_gf, self.value_sff = self.get_gf_sff(self.value_net)
        self.saver = tf.train.Saver(max_to_keep=5)
        ################## action net
        self.action_net.action_dist_stds_n = tf.exp(self.action_net.action_dist_logstds_n)
        self.dist_info_vars = dict(mean=self.action_net.action_dist_means_n, log_std=self.action_net.action_dist_logstds_n)
        self.likehood_action_dist = self.distribution.log_likelihood_sym(self.action_net.action_n, self.dist_info_vars)
        self.likehood_action_dist = tf.log(tf.clip_by_value(tf.exp(self.likehood_action_dist), 1e-20, 1.0))
        self.entropy = - tf.reduce_sum(tf.exp(self.likehood_action_dist) * self.likehood_action_dist)
        self.action_net.loss = - tf.reduce_sum(self.likehood_action_dist * self.action_net.advant) + self.entropy * self.pms.entropy_beta
        # self.action_net.theta_gradient = flatgrad(self.action_net.loss, self.action_net.var_list)
        self.action_net.optimizer = tf.train.AdamOptimizer()
        # self.action_net.theta_gradient_vars = self.action_net.optimizer.compute_gradients(self.action_net.loss, self.action_net.var_list)
        # self.action_net.gradients_value = [grad for (grad, var) in self.action_net.theta_gradient_vars]
        # self.action_net.gradient = tf.concat(0, [tf.reshape(grad, [np.prod(var_shape(var))])
        #                       for (grad, var) in zip(self.action_net.gradients_value, self.action_net.var_list)])
        # self.action_net.gradient = flatgrad(self.action_net.loss, self.action_net.var_list)
        self.action_net.optimizer_op = self.action_net.optimizer.minimize(self.action_net.loss)
        ################## value net
        self.value_net.loss = tf.nn.l2_loss(self.value_net.R - tf.reshape(self.value_net.value, [-1]))
        # self.value_net.theta_v_gradient = flatgrad(self.value_net.loss, self.value_net.var_list)
        self.value_net.optimizer = tf.train.AdamOptimizer()
        # self.value_net.theta_v_gradient_vars = self.value_net.optimizer.compute_gradients(self.value_net.loss, self.value_net.var_list)
        #
        # self.value_net.gradients_value = [grad for (grad, var) in self.value_net.theta_v_gradient_vars]
        # self.value_net.gradient = tf.concat(0, [tf.reshape(grad, [np.prod(var_shape(var))])
        #                                          for (grad, var) in
        #                                          zip(self.value_net.gradients_value, self.value_net.var_list)])
        # self.value_net.gradient = flatgrad(self.value_net.loss, self.value_net.var_list)
        self.value_net.optimizer_op = self.value_net.optimizer.minimize(self.value_net.loss)
        ################## initialize parameters
        self.session.run(tf.global_variables_initializer())

    def get_action(self, state):
        # action_dist_logstd = np.expand_dims([np.log(pms.std)], 0)
        action_dist_means_n, action_dist_stds_n = self.session.run(
            [self.action_net.action_dist_means_n, self.action_net.action_dist_stds_n],
            {self.action_net.states: state}
        )
        # print action_dist_stds_n
        if self.pms.train_flag:
            rnd = np.random.normal(size=action_dist_means_n[0].shape)
            action = rnd * action_dist_stds_n[0] + action_dist_means_n[0]
        else:
            action = action_dist_means_n[0]
        # action = np.clip(action, pms.min_a, pms.max_a)
        # print "train action: " + str(action)
        return action, action_dist_means_n[0], action_dist_stds_n[0]

    def get_value(self, state):
        value = self.session.run(self.value_net.value, feed_dict={self.value_net.states:state})
        return value

    def get_theta_gradient(self, advants, actions, states):
        # a list:[(grad, var), (grad, var)]
        # full_gradient = self.session.run(self.action_net.gradient,
        #                  feed_dict={self.action_net.advant:advants,
        #                             self.action_net.action_n:actions,
        #                             self.action_net.states:states})
        # return 1e-6 * full_gradient
        # first run optimizer_op for one time, and then the theta is change, get the new theta and
        # use new theta to minus old theta and get the gradient
        old_theta = self.action_gf()
        _, theta_loss = self.session.run([self.action_net.optimizer_op, self.action_net.loss], feed_dict={
            self.action_net.advant: advants,
            self.action_net.action_n:actions,
            self.action_net.states:states
        })
        new_theta = self.action_gf()
        print "theta loss:" + str(theta_loss)
        return new_theta - old_theta

    def get_theta_v_gradient(self, R, states):
        # a list:[(grad, var), (grad, var)]
        # full_gradient = self.session.run(self.value_net.gradient,
        #                         feed_dict={
        #                             self.value_net.R:R,
        #                             self.value_net.states:states
        #                         })
        # return 1e-6 * full_gradient
        old_theta = self.value_gf()
        self.session.run(self.value_net.optimizer_op, feed_dict={
            self.value_net.R: R,
            self.value_net.states:states
        })
        new_theta = self.value_gf()
        return new_theta - old_theta

    def rollout(self):
        # rollout a path
        rewards = []
        states = []
        actions = []
        means = []
        log_stds = []
        state = self.env.reset()
        for index in xrange(self.pms.max_path_length):
            if self.pms.render:
                self.env.render()
            states.append(state)
            action, mean, logstd = self.get_action([state])
            means.append(mean)
            log_stds.append(logstd)
            actions.append(action)
            state_next, reward, _, _ = self.env.step(action)
            rewards.append(reward)
            state = state_next
        path = dict(rewards=rewards, actions=actions, states=states, means=means, log_stds=log_stds)

        print "rewards:" + str(np.sum(np.array(path["rewards"])))
        return path

    def accumulate_gradient(self, path):
        states = path["states"]
        actions = path["actions"]
        rewards = path["rewards"]
        # returns shape : (self.pms.max_path_length,)
        returns = discount(np.array(rewards), self.pms.discount)
        # values shape : (self.pms.mat_path_length, 1)
        values = self.get_value(states)
        # print "values:" +str(values)
        # advant shape : (self.pms.mat_path_length,)
        advants = returns - np.concatenate(values, axis=0)
        if self.pms.center_adv:
            advants -= np.mean(advants)
            advants /= (advants.std() + 1e-8)
        self.delta_theta = self.get_theta_gradient(advants, actions, states)
        self.delta_theta_v = self.get_theta_v_gradient(returns, states)
        # print "value_loss:" + str(self.session.run(self.value_net.loss, feed_dict={
        #     self.value_net.R:returns,
        #     self.value_net.states:states}))
        # print "returns" + str(returns)
        # print "advants" + str(advants)
        # theta = self.action_gf() + self.delta_theta
        # theta_v = self.value_gf() + self.delta_theta_v
        # return theta, theta_v

    def asyc_gradient(self, action_net_param, value_net_param):
        # asyc network params
        action_net_param = action_net_param
        value_net_param = value_net_param
        self.action_sff(action_net_param)
        self.value_sff(value_net_param)

    def run(self):
        self.delta_theta = 0.0
        self.delta_theta_v = 0.0
        # initial network
        self.init_network()
        self.init_environment()
        while True:
            command = self.task_q.get()
            if command["type"] == "TRAIN":
                action_net_param = command["action_param"]
                value_net_param = command["value_param"]
                self.asyc_gradient(action_net_param, value_net_param)
                # rollout and get path dealing result
                path = self.rollout()
                # print "states" + str(path["states"])
                # print "actions" + str(path["actions"])
                # accumulate gradient

                self.accumulate_gradient(path)
                self.result_q.put((self.delta_theta, self.delta_theta_v))
            elif command["type"] == "GET_PARAM":
                theta = self.action_gf()
                theta_v = self.value_gf()
                self.result_q.put((theta, theta_v))
            elif command["type"] == "TEST":
                action_net_param = command["action_param"]
                value_net_param = command["value_param"]
                self.asyc_gradient(action_net_param, value_net_param)
                path = self.rollout()
            elif command["type"] == "STOP":
                break
            self.task_q.task_done()
        return
