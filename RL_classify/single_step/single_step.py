import os

import gym
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from RLToolbox.agent.TRPO_agent import TRPOAgent
from RLToolbox.storage.storage_continous import Storage
from RLToolbox.toolbox.baseline.baseline_zeros import Baseline
from RLToolbox.toolbox.distribution.diagonal_gaussian import DiagonalGaussian

from parameters import PMS_base
from RLToolbox.network.network import Network
from RLToolbox.network.AlexNet import AlexNet
from RLToolbox.network.Vgg16Net import NetworkTLVgg
from RL_classify.single_step.environment import EnvironmentClassify
from RL_classify.single_step.agent_classify import ClassifyAgent
import prettytensor as pt

class NetworkTL(Network):
    def __init__(self, scope, pms):
        super(NetworkTL , self).__init__(scope , pms)
        with tf.variable_scope("%s_shared" % scope):
            self.obs = obs = tf.placeholder(
                tf.float32 , shape=[None] + pms.obs_shape , name="%s_obs" % scope)
            self.action_n = tf.placeholder(tf.float32 , shape=[None , pms.action_shape] , name="%s_action" % scope)
            self.advant = tf.placeholder(tf.float32 , shape=[None] , name="%s_advant" % scope)
            self.old_dist_means_n = tf.placeholder(tf.float32 , shape=[None , pms.action_shape],
                                                   name="%s_oldaction_dist_means" % scope)
            self.old_dist_logstds_n = tf.placeholder(tf.float32 , shape=[None , pms.action_shape],
                                                     name="%s_oldaction_dist_logstds" % scope)
            network = tl.layers.InputLayer(self.obs, name='%s_input_layer'%scope)
            network = tl.layers.DenseLayer(network , n_units=128 ,W_init = tf.truncated_normal_initializer(stddev=0.1),
                                           act=tf.nn.relu , name="%s_fc1"%scope)
            network = tl.layers.DenseLayer(network , n_units=128 ,W_init = tf.truncated_normal_initializer(stddev=0.1),
                                           act=tf.nn.relu , name="%s_fc2"%scope)
            network = tl.layers.DenseLayer(network , n_units=64 ,W_init = tf.truncated_normal_initializer(stddev=0.1),
                                           act=tf.nn.relu , name="%s_fc3" % scope)
            network = tl.layers.DenseLayer(network , n_units=pms.action_shape,
                                          name="%s_fc4"%scope)
            self.action_dist_means_n = network.outputs
            self.N = tf.shape(obs)[0]
            Nf = tf.cast(self.N , tf.float32)
            self.action_dist_logstd_param = tf.Variable(
                (pms.std * np.random.randn(1 , pms.action_shape)).astype(np.float32) , name="%spolicy_logstd" % scope)
            self.action_dist_logstds_n = tf.tile(self.action_dist_logstd_param,
                                                 tf.pack((tf.shape(self.action_dist_means_n)[0] , 1)))
            self.var_list = [v for v in tf.trainable_variables() if v.name.startswith(scope)]
    def asyc_parameters(self, session=None):
        print "asyc parameters"

if __name__ == "__main__":
    if not os.path.isdir("./checkpoint"):
        os.makedirs("./checkpoint")
    if not os.path.isdir("./log"):
        os.makedirs("./log")
    from tensorflow.python.framework import ops
    from tensorflow.python.ops import array_ops
    @ops.RegisterGradient("MaxPoolGrad")
    def _MaxPoolGradGrad(op , grad):
        return (array_ops.zeros(shape=array_ops.shape(op.inputs[0]) , dtype=op.inputs[0].dtype) ,
                array_ops.zeros(shape=array_ops.shape(op.inputs[1]) , dtype=op.inputs[1].dtype) ,
                array_ops.ones(shape=array_ops.shape(op.inputs[2]) , dtype=op.inputs[2].dtype))
    pms = PMS_base()
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.per_process_gpu_memory_fraction = pms.GPU_fraction
    session = tf.Session(config=config)
    env = EnvironmentClassify(gym.make(pms.environment_name), pms=pms, session=session)
    net = NetworkTL("continous", pms=pms)
    a = [v.name for v in net.var_list]
    print "train variables:", a
    baseline = Baseline()
    storage = None
    distribution = DiagonalGaussian(pms.action_shape)
    agent = ClassifyAgent(env, session, baseline, storage, distribution, net, pms)
    agent.storage = Storage(agent , env , baseline, pms)
    env.agent = agent
    pms.train_flag = True
    if pms.train_flag:
        agent.learn()
    else:
        # agent.test_tracking(None, '/home/wyp/gym/gym/envs/object_tracker/datas/David3/img')
        # agent.test_gym_env(None)
        agent.test(None, load=True)
    if 'session' in locals() and session is not None:
        print('Close interactive session')
        session.close()
    # env.monitor.close()
    # gym.upload(training_dir,
    #            algorithm_id='trpo_ff')
