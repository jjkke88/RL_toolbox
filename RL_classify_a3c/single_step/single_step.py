import multiprocessing
import os
import time

import gym
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from RLToolbox.toolbox.common.utils import *
from RL_classify_a3c.single_step.classify_agent import ClassifyAgent
from RLToolbox.toolbox.distribution.diagonal_gaussian import DiagonalGaussian
from RL_classify_a3c.single_step.environment import EnvironmentClassify
from RL_classify_a3c.single_step.parameters import PMS_base


class NetworkTLAction(object):
    def __init__(self, scope):
        with tf.variable_scope("%s_shared" % scope):
            self.states = tf.placeholder(
                tf.float32 , shape=[None] + pms.obs_shape , name="%s_obs" % scope)
            self.action_n = tf.placeholder(tf.float32 , shape=[None , pms.action_shape] , name="%s_action" % scope)
            self.advant = tf.placeholder(tf.float32 , shape=[None] , name="%s_advant" % scope)

            network = tl.layers.InputLayer(self.states , name='%s_input_layer'%scope)
            network = tl.layers.DenseLayer(network , n_units=64 ,
                                           act=tf.nn.relu , name="%s_fc1"%scope)
            network = tl.layers.DenseLayer(network , n_units=64 ,
                                           act=tf.nn.relu , name="%s_fc2"%scope)
            network = tl.layers.DenseLayer(network , n_units=pms.action_shape,
                                           name="%s_fc3"%scope)
            self.action_dist_means_n = network.outputs

            self.action_dist_logstd_param = tf.Variable(
                (.01 * np.random.randn(1 , pms.action_shape)).astype(np.float32) , name="%spolicy_logstd" % scope)
            # self.action_dist_logstd_param = tf.maximum(self.action_dist_logstd_param, np.log(pms.min_std))
            self.action_dist_logstds_n = tf.tile(self.action_dist_logstd_param ,
                                                 tf.pack((tf.shape(self.action_dist_means_n)[0] , 1)))
            self.var_list = [v for v in tf.trainable_variables() if v.name.startswith(scope)]

class NetworkTLValue(object):
    def __init__(self, scope):
        with tf.variable_scope("%s_shared" % scope):
            self.states = tf.placeholder(
                tf.float32 , shape=[None] + pms.obs_shape , name="%s_obs" % scope)

            self.R = tf.placeholder(tf.float32 , shape=[None] , name="%s_R" % scope)

            network = tl.layers.InputLayer(self.states , name='%s_input_layer'%scope)
            network = tl.layers.DenseLayer(network , n_units=64,
                                           act=tf.nn.relu , name="%s_fc1"%scope)
            # network = tl.layers.DenseLayer(network , n_units=64 ,
            #                                act=tf.nn.relu , name="%s_fc2"%scope)
            network = tl.layers.DenseLayer(network , n_units=1,name="%s_fc3"%scope)
            self.value = network.outputs
            self.var_list = [v for v in tf.trainable_variables() if v.name.startswith(scope)]

class NetworkTLClassify(object):
    def __init__(self, scope):
        with tf.variable_scope("%s_shared" % scope):
            self.x = tf.placeholder(
                tf.float32 , shape=[None] + pms.obs_shape , name="%s_obs" % scope)
            self.y_ = tf.placeholder(tf.int32, shape=[None, ], name='%s_y_'%scope)
            network = tl.layers.InputLayer(self.x , name='%s_input_layer'%scope)
            network = tl.layers.DenseLayer(network , n_units=64,
                                           act=tf.nn.relu , name="%s_fc1"%scope)
            network = tl.layers.DenseLayer(network , n_units=64 ,
                                           act=tf.nn.relu , name="%s_fc2"%scope)
            network = tl.layers.DenseLayer(network , n_units=pms.class_number, name="%s_fc3"%scope)
            self.y = network.outputs
            self.probs = tf.nn.softmax(self.y)
            self.correct_prediction = tf.equal(tf.cast(tf.argmax(self.y, 1), tf.float32),
                                                            tf.cast(self.y_, tf.float32))
            self.cost = tl.cost.cross_entropy(self.y, self.y_, name='cost')
            self.acc = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            self.var_list = [v for v in tf.trainable_variables() if v.name.startswith(scope)]

if __name__ == "__main__":
    if not os.path.isdir("./checkpoint"):
        os.makedirs("./checkpoint")
    if not os.path.isdir("./log"):
        os.makedirs("./log")
    if not os.path.isdir("./test_image_dir"):
        os.makedirs("./test_image_dir")
    params = {"environment":EnvironmentClassify,
              "distribution":DiagonalGaussian,
              "agent":ClassifyAgent}
    pms = PMS_base()
    pms.train_flag = False
    # pms.render = True
    args = pms
    # args.max_path_length = gym.spec(args.environment_name).timestep_limit
    learner_tasks = multiprocessing.JoinableQueue()
    learner_results = multiprocessing.Queue()
    learner_env = params["environment"]
    net = dict(action_net=NetworkTLAction("action"), value_net=NetworkTLValue("value"), classify_net=NetworkTLClassify("classify"))
    # baseline = params["baseline"]()
    distribution = params["distribution"](pms.action_shape)
    learners = []
    for i in xrange(pms.jobs):
        learner = params["agent"](env=learner_env, session=None, baseline=None, storage=None, distribution=distribution, net=net, pms=pms, task_q=learner_tasks, result_q=learner_results, process_id=i)
        learners.append(learner)
    for learner in learners:
        learner.start()
    learner_tasks.put(dict(type="GET_PARAM"))
    learner_tasks.join()
    theta, theta_v, theta_c = learner_results.get()
    if pms.train_flag:
        # initial net parameters
        for i in xrange(10000):
            if i % pms.save_model_times == 0 and i != 0:
                for k in xrange(pms.jobs):
                    command = dict(type="SAVE_PARAM")
                    learner_tasks.put(command)
                learner_tasks.join()
            if i % pms.test_frequency == 0 and i != 0:
                for k in xrange(pms.jobs):
                    command = dict(type="TEST")
                    learner_tasks.put(command)
                learner_tasks.join()
                # print "test result size:%d" % learner_results.qsize()
                test_result = learner_results.get()
                print "test_result:" + str(test_result)
                # print "test result end size:%d" % learner_results.qsize()

            for k in xrange(pms.jobs):
                command = dict(type="TRAIN", action_param=theta, value_param=theta_v, classify_param=theta_c)
                learner_tasks.put(command)
            learner_tasks.join()
            thetas = []
            theta_vs = []
            theta_cs = []
            # print "train result size:%d" % learner_results.qsize()
            for k in xrange(pms.jobs):
                delta_theta, delta_theta_v, delta_theta_c = learner_results.get()
                thetas.append(delta_theta)
                theta_vs.append(delta_theta_v)
                theta_cs.append(delta_theta_c)
            # print "train result end size:%d" % learner_results.qsize()
            # update net
            theta += np.array(thetas).sum(axis=0)
            theta_v += np.array(theta_vs).sum(axis=0)
            theta_c += np.array(theta_cs).sum(axis=0)
            # print "theta:" + str(theta)
            # print "theta_v" + str(theta_v)
    else:
        data = np.load(os.path.join(pms.checkpoint_dir, "model.npz"))
        theta = data["theta"]
        theta_v = data["theta_v"]
        theta_c = data["theta_c"]
        for k in xrange(pms.jobs):
            command = dict(type="SET_PARAM", action_param=theta, value_param=theta_v, classify_param=theta_c)
            learner_tasks.put(command)
        learner_tasks.join()
        for k in xrange(pms.jobs):
            command = dict(type="TEST")
            learner_tasks.put(command)
        learner_tasks.join()
        # print "test result size:%d" % learner_results.qsize()
        test_result = learner_results.get()
        print "test_result:" + str(test_result)
            # print "test result end size:%d" % learner_results.qsize()
    exit()