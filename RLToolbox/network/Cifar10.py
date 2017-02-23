from RLToolbox.network.network import Network
import tensorflow as tf
import tensorlayer as tl
import numpy as np


class Cifar10Net(Network):
    def __init__(self, scope, pms):
        super(Cifar10Net, self).__init__(scope, pms)
        with tf.variable_scope("%s_shared" % scope):
            self.obs = obs = tf.placeholder(
                tf.float32 , shape=[None] + [pms.obs_shape[0] , pms.obs_shape[1] , pms.obs_shape[2]] ,
                name="%s_obs" % scope)
            self.obs_target = obs_target = tf.placeholder(
                tf.float32 , shape=[None] + pms.obs_shape , name="%s_obs_target" % scope)
            self.action_n = tf.placeholder(tf.float32 , shape=[None , pms.action_shape] , name="%s_action" % scope)
            self.advant = tf.placeholder(tf.float32 , shape=[None] , name="%s_advant" % scope)
            self.old_dist_means_n = tf.placeholder(tf.float32 , shape=[None , pms.action_shape] ,
                                                   name="%s_oldaction_dist_means" % scope)
            self.old_dist_logstds_n = tf.placeholder(tf.float32 , shape=[None , pms.action_shape] ,
                                                     name="%s_oldaction_dist_logstds" % scope)

            network = tl.layers.InputLayer(self.obs , name='input_layer')
            network = tl.layers.Conv2dLayer(network ,
                                            act=tf.nn.relu ,
                                            shape=[5 , 5 , 3 , 64] ,  # 64 features for each 5x5x3 patch
                                            strides=[1 , 1 , 1 , 1] ,
                                            padding='SAME' ,
                                            W_init=tf.truncated_normal_initializer(stddev=5e-2) ,
                                            W_init_args={} ,
                                            b_init=tf.constant_initializer(value=0.0) ,
                                            b_init_args={} ,
                                            name='cnn_layer1')  # output: (batch_size, 24, 24, 64)
            network = tl.layers.PoolLayer(network ,
                                          ksize=[1 , 3 , 3 , 1] ,
                                          strides=[1 , 2 , 2 , 1] ,
                                          padding='SAME' ,
                                          pool=tf.nn.max_pool ,
                                          name='pool_layer1' , )  # output: (batch_size, 12, 12, 64)
            network.outputs = tf.nn.lrn(network.outputs , 4 , bias=1.0 , alpha=0.001 / 9.0 ,
                                        beta=0.75 , name='norm1')
            network = tl.layers.Conv2dLayer(network ,
                                            act=tf.nn.relu ,
                                            shape=[5 , 5 , 64 , 64] ,  # 64 features for each 5x5 patch
                                            strides=[1 , 1 , 1 , 1] ,
                                            padding='SAME' ,
                                            W_init=tf.truncated_normal_initializer(stddev=5e-2) ,
                                            W_init_args={} ,
                                            b_init=tf.constant_initializer(value=0.1) ,
                                            b_init_args={} ,
                                            name='cnn_layer2')  # output: (batch_size, 12, 12, 64)
            network.outputs = tf.nn.lrn(network.outputs , 4 , bias=1.0 , alpha=0.001 / 9.0 ,
                                        beta=0.75 , name='norm2')
            network = tl.layers.PoolLayer(network ,
                                          ksize=[1 , 3 , 3 , 1] ,
                                          strides=[1 , 2 , 2 , 1] ,
                                          padding='SAME' ,
                                          pool=tf.nn.max_pool ,
                                          name='pool_layer2')  # output: (batch_size, 6, 6, 64)
            network = tl.layers.FlattenLayer(network , name='flatten_layer')  # output: (batch_size, 2304)
            network = tl.layers.DenseLayer(network , n_units=384 , act=tf.nn.relu ,
                                           W_init=tf.truncated_normal_initializer(stddev=0.04) ,
                                           W_init_args={} ,
                                           b_init=tf.constant_initializer(value=0.1) ,
                                           b_init_args={} , name='relu1')  # output: (batch_size, 384)
            network = tl.layers.DenseLayer(network , n_units=192 , act=tf.nn.relu ,
                                           W_init=tf.truncated_normal_initializer(stddev=0.04) ,
                                           W_init_args={} ,
                                           b_init=tf.constant_initializer(value=0.1) ,
                                           b_init_args={} , name='relu2')  # output: (batch_size, 192)
            network = tl.layers.DenseLayer(network , n_units=10 , act=tf.identity ,
                                           W_init=tf.truncated_normal_initializer(stddev=1 / 192.0) ,
                                           W_init_args={} ,
                                           b_init=tf.constant_initializer(value=0.0) ,
                                           name='output_layer')  # output: (batch_size, 10)
            y = network.outputs