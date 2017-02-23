import multiprocessing
import os
import time

import gym
import numpy as np
import prettytensor as pt
import tensorflow as tf
import tensorlayer as tl
from RLToolbox.agent.TRPO_parallel_agent import TRPOParallelAgent
from RLToolbox.storage.storage_continous_parallel import ParallelStorage
from RLToolbox.toolbox.baseline.baseline_zeros import Baseline
from RLToolbox.toolbox.distribution.diagonal_gaussian import DiagonalGaussian
from RLToolbox.network.Vgg16Net import NetworkTLVgg
from environment import EnvironmentTracking
from RLToolbox.network.AlexNet import AlexNet

from parameters import PMS_base
from RLToolbox.network.network import Network

class NetworkContinous(object):
    def __init__(self, scope):
        with tf.variable_scope("%s_shared" % scope):
            self.obs = obs = tf.placeholder(
                tf.float32, shape=[None] + pms.obs_shape, name="%s_obs"%scope)
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
class NetworkTL(Network):
    def __init__(self, scope, pms):
        super(NetworkTL , self).__init__(scope , pms)
        with tf.variable_scope("%s_shared" % scope):
            self.obs = obs = tf.placeholder(
                tf.float32 , shape=[None] + pms.obs_shape , name="%s_obs" % scope)
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
    def asyc_parameters(self, session=None):
        print "asyc parameters"



if __name__ == "__main__":
    if not os.path.isdir("./checkpoint"):
        os.makedirs("./checkpoint")
    if not os.path.isdir("./log"):
        os.makedirs("./log")
    params = {"environment":EnvironmentTracking,
              "network":NetworkTL,
              "baseline":Baseline,
              "distribution":DiagonalGaussian,
              "storage":ParallelStorage,
              "agent":TRPOParallelAgent}

    from tensorflow.python.framework import ops
    from tensorflow.python.ops import array_ops
    @ops.RegisterGradient("MaxPoolGrad")
    def _MaxPoolGradGrad(op , grad):
        return (array_ops.zeros(shape=array_ops.shape(op.inputs[0]) , dtype=op.inputs[0].dtype) ,
                array_ops.zeros(shape=array_ops.shape(op.inputs[1]) , dtype=op.inputs[1].dtype) ,
                array_ops.ones(shape=array_ops.shape(op.inputs[2]) , dtype=op.inputs[2].dtype))
    pms = PMS_base()
    pms.train_flag = False
    pms.render = True
    args = pms
    args.max_pathlength = gym.spec(args.environment_name).timestep_limit
    learner_tasks = multiprocessing.JoinableQueue()
    learner_results = multiprocessing.Queue()
    learner_env = params["environment"](gym.make(args.environment_name), pms=pms)
    net = params["network"]("continous", pms)
    baseline = params["baseline"]()
    distribution = params["distribution"](pms.action_shape)
    rollouts = params["storage"](None, None, baseline=baseline, pms=pms, net_class=params["network"], env_class=params['environment'])
    learner = params["agent"](learner_env, session=None, baseline=baseline, storage=rollouts, distribution=distribution, net=net, pms=pms, task_q=learner_tasks , result_q=learner_results)
    learner.start()
    learner_tasks.put(1)
    learner_tasks.join()
    starting_weights = learner_results.get()
    rollouts.set_policy_weights(starting_weights)

    start_time = time.time()
    history = {}
    history["rollout_time"] = []
    history["learn_time"] = []
    history["mean_reward"] = []
    history["timesteps"] = []

    # start it off with a big negative number
    last_reward = -1000000
    recent_total_reward = 0

    if pms.train_flag is True:
        for iteration in xrange(args.max_iter_number):
            # runs a bunch of async processes that collect rollouts
            paths = rollouts.get_paths()
            # Why is the learner in an async process?
            # Well, it turns out tensorflow has an issue: when there's a tf.Session in the main thread
            # and an async process creates another tf.Session, it will freeze up.
            # To solve this, we just make the learner's tf.Session in its own async process,
            # and wait until the learner's done before continuing the main thread.
            learn_start = time.time()
            if iteration%pms.save_model_times == 0:
                learner_tasks.put((2 , args.max_kl, 1, iteration))
            else:
                learner_tasks.put((2, args.max_kl, 0, iteration))
            learner_tasks.put(paths)
            learner_tasks.join()

            stats , theta , thprev = learner_results.get()
            learn_time = (time.time() - learn_start) / 60.0
            print
            print "-------- Iteration %d ----------" % iteration
            # print "Total time: %.2f mins" % ((time.time() - start_time) / 60.0)
            #
            # history["rollout_time"].append(rollout_time)
            # history["learn_time"].append(learn_time)
            # history["mean_reward"].append(mean_reward)
            # history["timesteps"].append(args.timesteps_per_batch)
            for k , v in stats.iteritems():
                print(k + ": " + " " * (40 - len(k)) + str(v))
            recent_total_reward += stats["Average sum of rewards per episode"]

            if args.decay_method == "adaptive":
                if iteration % 10 == 0:
                    if recent_total_reward < last_reward:
                        print "Policy is not improving. Decrease KL and increase steps."
                        if args.max_kl > 0.001:
                            args.max_kl -= args.kl_adapt
                    else:
                        print "Policy is improving. Increase KL and decrease steps."
                        if args.max_kl < 0.01:
                            args.max_kl += args.kl_adapt
                    last_reward = recent_total_reward
                    recent_total_reward = 0

            if args.decay_method == "linear":
                if args.max_kl > 0.001:
                    args.max_kl -= args.kl_adapt

            if args.decay_method == "exponential":
                if args.max_kl > 0.001:
                    args.max_kl *= args.kl_adapt
            rollouts.set_policy_weights(theta)
    else:
        from agent.TRPO_agent import TRPOAgent
        from environment.gym_environment import Environment
        from storage.storage_continous import Storage
        session = tf.Session()
        baseline = Baseline()
        storage = None
        pms = PMS_base().pms
        # pms.train_flag = False
        # pms.render = True
        env = Environment(gym.make(pms.environment_name) , pms=pms)
        distribution = DiagonalGaussian(pms.action_shape)
        agent = TRPOAgent(env , session , baseline , storage , distribution , net , pms)
        agent.storage = Storage(agent , env , baseline , pms)
        agent.test(pms.checkpoint_file)


    rollouts.end()
