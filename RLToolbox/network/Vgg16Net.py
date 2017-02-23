from RLToolbox.network.network import Network
import tensorflow as tf
import tensorlayer as tl
import numpy as np

class NetworkTLVgg(Network):
    def __init__(self, scope, pms):
        super(NetworkTLVgg, self).__init__(scope, pms)
        with tf.variable_scope("%s_shared" % scope):
            self.obs = obs = tf.placeholder(
                tf.float32 , shape=[None] + [pms.obs_shape[0], pms.obs_shape[1], pms.obs_shape[2]] , name="%s_obs" % scope)
            self.obs_target = obs_target = tf.placeholder(
                tf.float32 , shape=[None] + pms.obs_shape , name="%s_obs_target" % scope)
            self.action_n = tf.placeholder(tf.float32 , shape=[None, pms.action_shape] , name="%s_action" % scope)
            self.advant = tf.placeholder(tf.float32 , shape=[None] , name="%s_advant" % scope)
            self.old_dist_means_n = tf.placeholder(tf.float32 , shape=[None , pms.action_shape] ,
                                                   name="%s_oldaction_dist_means" % scope)
            self.old_dist_logstds_n = tf.placeholder(tf.float32 , shape=[None , pms.action_shape] ,
                                                     name="%s_oldaction_dist_logstds" % scope)

            network = tl.layers.InputLayer(self.obs , name='%s_input_layer'%scope)
            net_cnn = self.conv_layers(network)
            network = tl.layers.FlattenLayer(net_cnn, name="%s_flatten_layer"%scope)

            # network_target = tl.layers.InputLayer(self.obs , name='%s_target_input_layer'%scope)
            # network_target_cnn = self.conv_layers(network_target)
            # network_target = tl.layers.FlattenLayer(network_target_cnn, name="%s_target_flatten_layer"%scope)
            # net = tl.layers.ConcatLayer(layer=[network , network_target] , name='%s_concat_layer'%scope)
            self.net = self.fc_layers(network)

            self.action_dist_means_n = self.net.outputs
            self.N = tf.shape(obs)[0]
            self.action_dist_logstd_param = tf.Variable(
                (.01 * np.random.randn(1 , pms.action_shape)).astype(np.float32) , name="%spolicy_logstd" % scope)
            # self.action_dist_logstd_param = tf.maximum(self.action_dist_logstd_param, np.log(pms.min_std))
            self.action_dist_logstds_n = tf.tile(self.action_dist_logstd_param ,
                                                 tf.pack((tf.shape(self.action_dist_means_n)[0] , 1)))
            self.var_train_name = ['%s_shared/%s_fc4_relu' % (self.scope , self.scope) ,
                                   '%s_shared/%s_fc5_relu' % (self.scope , self.scope) ,
                                   '%s_shared/%s_fc6_relu' % (self.scope , self.scope)]
            start_name = "%s_shared/%s_fc_train" % (self.scope, self.scope)
            self.var_list = [v for v in tf.trainable_variables() if v.name.startswith(start_name)]

    def conv_layers(self, net_in):
        with tf.name_scope(self.scope) as scope:
            """
            Notice that we include a preprocessing layer that takes the RGB image
            with pixels values in the range of 0-255 and subtracts the mean image
            values (calculated over the entire ImageNet training set).
            """
            mean = tf.constant([123.68 , 116.779 , 103.939] , dtype=tf.float32 , shape=[1 , 1 , 1 , 3] ,
                               name='img_mean')
            net_in.outputs = net_in.outputs - mean

        """ conv1 """
        network = tl.layers.Conv2dLayer(net_in ,
                                        act=tf.nn.relu ,
                                        shape=[3 , 3 , 3 , 64] ,  # 64 features for each 3x3 patch
                                        strides=[1 , 1 , 1 , 1] ,
                                        padding='SAME' ,
                                        name='%s_conv1_1'%self.scope)
        network = tl.layers.Conv2dLayer(network ,
                                        act=tf.nn.relu ,
                                        shape=[3 , 3 , 64 , 64] ,  # 64 features for each 3x3 patch
                                        strides=[1 , 1 , 1 , 1] ,
                                        padding='SAME' ,
                                        name='%s_conv1_2'%self.scope)
        network = tl.layers.PoolLayer(network ,
                                      ksize=[1 , 2 , 2 , 1] ,
                                      strides=[1 , 2 , 2 , 1] ,
                                      padding='SAME' ,
                                      pool=tf.nn.max_pool ,
                                      name='%s_pool1'%self.scope)
        """ conv2 """
        network = tl.layers.Conv2dLayer(network ,
                                        act=tf.nn.relu ,
                                        shape=[3 , 3 , 64 , 128] ,  # 128 features for each 3x3 patch
                                        strides=[1 , 1 , 1 , 1] ,
                                        padding='SAME' ,
                                        name='%s_conv2_1'%self.scope)
        network = tl.layers.Conv2dLayer(network ,
                                        act=tf.nn.relu ,
                                        shape=[3 , 3 , 128 , 128] ,  # 128 features for each 3x3 patch
                                        strides=[1 , 1 , 1 , 1] ,
                                        padding='SAME' ,
                                        name='%s_conv2_2'%self.scope)
        network = tl.layers.PoolLayer(network ,
                                      ksize=[1 , 2 , 2 , 1] ,
                                      strides=[1 , 2 , 2 , 1] ,
                                      padding='SAME' ,
                                      pool=tf.nn.max_pool ,
                                      name='%s_pool2'%self.scope)
        """ conv3 """
        network = tl.layers.Conv2dLayer(network ,
                                        act=tf.nn.relu ,
                                        shape=[3 , 3 , 128 , 256] ,  # 256 features for each 3x3 patch
                                        strides=[1 , 1 , 1 , 1] ,
                                        padding='SAME' ,
                                        name='%s_conv3_1'%self.scope)
        network = tl.layers.Conv2dLayer(network ,
                                        act=tf.nn.relu ,
                                        shape=[3 , 3 , 256 , 256] ,  # 256 features for each 3x3 patch
                                        strides=[1 , 1 , 1 , 1] ,
                                        padding='SAME' ,
                                        name='%s_conv3_2'%self.scope)
        network = tl.layers.Conv2dLayer(network ,
                                        act=tf.nn.relu ,
                                        shape=[3 , 3 , 256 , 256] ,  # 256 features for each 3x3 patch
                                        strides=[1 , 1 , 1 , 1] ,
                                        padding='SAME' ,
                                        name='%s_conv3_3'%self.scope)
        network = tl.layers.PoolLayer(network ,
                                      ksize=[1 , 2 , 2 , 1] ,
                                      strides=[1 , 2 , 2 , 1] ,
                                      padding='SAME' ,
                                      pool=tf.nn.max_pool ,
                                      name='%s_pool3'%self.scope)
        """ conv4 """
        network = tl.layers.Conv2dLayer(network ,
                                        act=tf.nn.relu ,
                                        shape=[3 , 3 , 256 , 512] ,  # 512 features for each 3x3 patch
                                        strides=[1 , 1 , 1 , 1] ,
                                        padding='SAME' ,
                                        name='%s_conv4_1'%self.scope)
        network = tl.layers.Conv2dLayer(network ,
                                        act=tf.nn.relu ,
                                        shape=[3 , 3 , 512 , 512] ,  # 512 features for each 3x3 patch
                                        strides=[1 , 1 , 1 , 1] ,
                                        padding='SAME' ,
                                        name='%s_conv4_2'%self.scope)
        network = tl.layers.Conv2dLayer(network ,
                                        act=tf.nn.relu ,
                                        shape=[3 , 3 , 512 , 512] ,  # 512 features for each 3x3 patch
                                        strides=[1 , 1 , 1 , 1] ,
                                        padding='SAME' ,
                                        name='%s_conv4_3'%self.scope)
        network = tl.layers.PoolLayer(network ,
                                      ksize=[1 , 2 , 2 , 1] ,
                                      strides=[1 , 2 , 2 , 1] ,
                                      padding='SAME' ,
                                      pool=tf.nn.max_pool ,
                                      name='%s_pool4'%self.scope)
        """ conv5 """
        network = tl.layers.Conv2dLayer(network ,
                                        act=tf.nn.relu ,
                                        shape=[3 , 3 , 512 , 512] ,  # 512 features for each 3x3 patch
                                        strides=[1 , 1 , 1 , 1] ,
                                        padding='SAME' ,
                                        name='%s_conv5_1'%self.scope)
        network = tl.layers.Conv2dLayer(network ,
                                        act=tf.nn.relu ,
                                        shape=[3 , 3 , 512 , 512] ,  # 512 features for each 3x3 patch
                                        strides=[1 , 1 , 1 , 1] ,
                                        padding='SAME' ,
                                        name='%s_conv5_2'%self.scope)
        network = tl.layers.Conv2dLayer(network ,
                                        act=tf.nn.relu ,
                                        shape=[3 , 3 , 512 , 512] ,  # 512 features for each 3x3 patch
                                        strides=[1 , 1 , 1 , 1] ,
                                        padding='SAME' ,
                                        name='%s_conv5_3'%self.scope)
        network = tl.layers.PoolLayer(network ,
                                      ksize=[1 , 2 , 2 , 1] ,
                                      strides=[1 , 2 , 2 , 1] ,
                                      padding='SAME' ,
                                      pool=tf.nn.max_pool ,
                                      name='%s_pool5'%self.scope)
        return network

    def fc_layers(self, net):
        network = tl.layers.FlattenLayer(net , name='%s_flatten'%self.scope)
        network = tl.layers.DenseLayer(network , n_units=4096,
                                       act=tf.nn.relu ,
                                       name='%s_fc1_relu'%self.scope)
        network = tl.layers.DenseLayer(network , n_units=4096 ,
                                       act=tf.nn.relu ,
                                       name='%s_fc2_relu'%self.scope)
        network = tl.layers.DenseLayer(network , n_units=1000 ,
                                       act=tf.nn.relu ,
                                       name='%s_fc3_relu' % self.scope)

        network = tl.layers.DenseLayer(network , n_units=128 ,
                                       act=tf.nn.relu ,
                                       name='%s_fc_train4_relu' % self.scope)
        network = tl.layers.DenseLayer(network , n_units=64 ,
                                       act=tf.nn.relu ,
                                       name='%s_fc_train5_relu' % self.scope)
        network = tl.layers.DenseLayer(network , n_units=pms.action_shape ,
                                       name='%s_fc_train6' % self.scope)
        return network

    def asyc_parameters(self, session):
        print "asyc_parameters"
        npz = np.load('vgg16_weights.npz')
        params = []
        for val in sorted(npz.items()):
                print("  Loading %s" % str(val[0]))
                params.append(val[1])

        tl.files.assign_params(session, params, self.net)