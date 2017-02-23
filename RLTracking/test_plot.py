import tensorlayer as tl
import  tensorflow as tf
import numpy as np
from parameters import PMS_base
class NetworkTLImage(object):
    def __init__(self, scope):
        with tf.variable_scope("%s_shared" % scope):
            self.obs = obs = tf.placeholder(
                tf.float32 , shape=[None] + pms.obs_shape , name="%s_obs" % scope)
            self.action_n = tf.placeholder(tf.float32 , shape=[None, pms.action_shape] , name="%s_action" % scope)
            self.advant = tf.placeholder(tf.float32 , shape=[None] , name="%s_advant" % scope)
            self.old_dist_means_n = tf.placeholder(tf.float32 , shape=[None , pms.action_shape] ,
                                                   name="%s_oldaction_dist_means" % scope)
            self.old_dist_logstds_n = tf.placeholder(tf.float32 , shape=[None , pms.action_shape] ,
                                                     name="%s_oldaction_dist_logstds" % scope)
            network = tl.layers.InputLayer(self.obs , name='%s_input_layer'%scope)
            network = tl.layers.Conv2dLayer(network ,
                                            act=tf.nn.relu ,
                                            shape=[3 , 3 , 3 , 64] ,  # 64 features for each 3x3 patch
                                            strides=[1 , 1 , 1 , 1] ,
                                            padding='SAME' ,
                                            name='%s_conv1'%scope)
            network = tl.layers.FlattenLayer(network , name='%s_flatten'%scope)
            network = tl.layers.DenseLayer(network , n_units=64 ,
                                           act=tf.nn.relu , name="%s_fc1"%scope)
            network = tl.layers.DenseLayer(network , n_units=64 ,
                                           act=tf.nn.relu , name="%s_fc2"%scope)
            network = tl.layers.DenseLayer(network , n_units=pms.action_shape ,
                                           act=tf.nn.relu , name="%s_fc3"%scope)
            self.action_dist_means_n = network.outputs
            self.N = tf.shape(obs)[0]
            self.action_dist_logstd_param = tf.Variable(
                (.01 * np.random.randn(1 , pms.action_shape)).astype(np.float32) , name="%spolicy_logstd" % scope)
            # self.action_dist_logstd_param = tf.maximum(self.action_dist_logstd_param, np.log(pms.min_std))
            self.action_dist_logstds_n = tf.tile(self.action_dist_logstd_param ,
                                                 tf.pack((tf.shape(self.action_dist_means_n)[0] , 1)))
            self.var_list = [v for v in tf.trainable_variables() if v.name.startswith(scope)]

if __name__ == "__main__":
    # pms = PMS_base()
    # pms.obs_shape =[100, 100, 3]
    # net = NetworkTLImage("d")
    # session = tf.Session()
    # session.run(tf.initialize_all_variables())
    # tl.visualize.CNN2d(net.var_list[0].eval(session=session) , second=10 , saveable=False , name='cnn1_mnist' , fig_idx=2012)
    from RLToolbox.environment.gym_environment import Environment
    import gym
    from parameters import PMS_base
    pms = PMS_base()
    env = Environment(gym.make("ObjectTracker-v1"), pms=pms)
    obs = env.reset()
    env.render()
    print obs.shape