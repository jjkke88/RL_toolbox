import tensorlayer as tl
import tensorflow as tf
import numpy as np

class VggNet:
    def __init__(self, session, pms, scope="VGG"):
        self.session = session
        self.scope = scope
        self.pms = pms
        with tf.variable_scope("%s_shared" % scope):
            self.obs = obs = tf.placeholder(
                tf.float32 , shape=[None, 224, 224, 3],
                name="%s_obs" % scope)
            network = tl.layers.InputLayer(self.obs , name='%s_input_layer' % scope)
            net_cnn = self.conv_layers(network)
            network = tl.layers.FlattenLayer(net_cnn , name="%s_flatten_layer" % scope)
            self.net = self.fc_layers(network)
            self.feature_vector = self.net.outputs
            self.asyc_parameters(self.session)

    def conv_layers(self , net_in):
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
                                        name='%s_conv1_1' % self.scope)
        network = tl.layers.Conv2dLayer(network ,
                                        act=tf.nn.relu ,
                                        shape=[3 , 3 , 64 , 64] ,  # 64 features for each 3x3 patch
                                        strides=[1 , 1 , 1 , 1] ,
                                        padding='SAME' ,
                                        name='%s_conv1_2' % self.scope)
        network = tl.layers.PoolLayer(network ,
                                      ksize=[1 , 2 , 2 , 1] ,
                                      strides=[1 , 2 , 2 , 1] ,
                                      padding='SAME' ,
                                      pool=tf.nn.max_pool ,
                                      name='%s_pool1' % self.scope)
        """ conv2 """
        network = tl.layers.Conv2dLayer(network ,
                                        act=tf.nn.relu ,
                                        shape=[3 , 3 , 64 , 128] ,  # 128 features for each 3x3 patch
                                        strides=[1 , 1 , 1 , 1] ,
                                        padding='SAME' ,
                                        name='%s_conv2_1' % self.scope)
        network = tl.layers.Conv2dLayer(network ,
                                        act=tf.nn.relu ,
                                        shape=[3 , 3 , 128 , 128] ,  # 128 features for each 3x3 patch
                                        strides=[1 , 1 , 1 , 1] ,
                                        padding='SAME' ,
                                        name='%s_conv2_2' % self.scope)
        network = tl.layers.PoolLayer(network ,
                                      ksize=[1 , 2 , 2 , 1] ,
                                      strides=[1 , 2 , 2 , 1] ,
                                      padding='SAME' ,
                                      pool=tf.nn.max_pool ,
                                      name='%s_pool2' % self.scope)
        """ conv3 """
        network = tl.layers.Conv2dLayer(network ,
                                        act=tf.nn.relu ,
                                        shape=[3 , 3 , 128 , 256] ,  # 256 features for each 3x3 patch
                                        strides=[1 , 1 , 1 , 1] ,
                                        padding='SAME' ,
                                        name='%s_conv3_1' % self.scope)
        network = tl.layers.Conv2dLayer(network ,
                                        act=tf.nn.relu ,
                                        shape=[3 , 3 , 256 , 256] ,  # 256 features for each 3x3 patch
                                        strides=[1 , 1 , 1 , 1] ,
                                        padding='SAME' ,
                                        name='%s_conv3_2' % self.scope)
        network = tl.layers.Conv2dLayer(network ,
                                        act=tf.nn.relu ,
                                        shape=[3 , 3 , 256 , 256] ,  # 256 features for each 3x3 patch
                                        strides=[1 , 1 , 1 , 1] ,
                                        padding='SAME' ,
                                        name='%s_conv3_3' % self.scope)
        network = tl.layers.PoolLayer(network ,
                                      ksize=[1 , 2 , 2 , 1] ,
                                      strides=[1 , 2 , 2 , 1] ,
                                      padding='SAME' ,
                                      pool=tf.nn.max_pool ,
                                      name='%s_pool3' % self.scope)
        """ conv4 """
        network = tl.layers.Conv2dLayer(network ,
                                        act=tf.nn.relu ,
                                        shape=[3 , 3 , 256 , 512] ,  # 512 features for each 3x3 patch
                                        strides=[1 , 1 , 1 , 1] ,
                                        padding='SAME' ,
                                        name='%s_conv4_1' % self.scope)
        network = tl.layers.Conv2dLayer(network ,
                                        act=tf.nn.relu ,
                                        shape=[3 , 3 , 512 , 512] ,  # 512 features for each 3x3 patch
                                        strides=[1 , 1 , 1 , 1] ,
                                        padding='SAME' ,
                                        name='%s_conv4_2' % self.scope)
        network = tl.layers.Conv2dLayer(network ,
                                        act=tf.nn.relu ,
                                        shape=[3 , 3 , 512 , 512] ,  # 512 features for each 3x3 patch
                                        strides=[1 , 1 , 1 , 1] ,
                                        padding='SAME' ,
                                        name='%s_conv4_3' % self.scope)
        network = tl.layers.PoolLayer(network ,
                                      ksize=[1 , 2 , 2 , 1] ,
                                      strides=[1 , 2 , 2 , 1] ,
                                      padding='SAME' ,
                                      pool=tf.nn.max_pool ,
                                      name='%s_pool4' % self.scope)
        """ conv5 """
        network = tl.layers.Conv2dLayer(network ,
                                        act=tf.nn.relu ,
                                        shape=[3 , 3 , 512 , 512] ,  # 512 features for each 3x3 patch
                                        strides=[1 , 1 , 1 , 1] ,
                                        padding='SAME' ,
                                        name='%s_conv5_1' % self.scope)
        network = tl.layers.Conv2dLayer(network ,
                                        act=tf.nn.relu ,
                                        shape=[3 , 3 , 512 , 512] ,  # 512 features for each 3x3 patch
                                        strides=[1 , 1 , 1 , 1] ,
                                        padding='SAME' ,
                                        name='%s_conv5_2' % self.scope)
        network = tl.layers.Conv2dLayer(network ,
                                        act=tf.nn.relu ,
                                        shape=[3 , 3 , 512 , 512] ,  # 512 features for each 3x3 patch
                                        strides=[1 , 1 , 1 , 1] ,
                                        padding='SAME' ,
                                        name='%s_conv5_3' % self.scope)
        network = tl.layers.PoolLayer(network ,
                                      ksize=[1 , 2 , 2 , 1] ,
                                      strides=[1 , 2 , 2 , 1] ,
                                      padding='SAME' ,
                                      pool=tf.nn.max_pool ,
                                      name='%s_pool5' % self.scope)
        return network


    def fc_layers(self , net):
        network = tl.layers.FlattenLayer(net , name='%s_flatten' % self.scope)
        network = tl.layers.DenseLayer(network , n_units=4096 ,
                                       act=tf.nn.relu ,
                                       name='%s_fc1_relu' % self.scope)
        network = tl.layers.DenseLayer(network , n_units=4096 ,
                                       act=tf.nn.relu ,
                                       name='%s_fc2_relu' % self.scope)
        network = tl.layers.DenseLayer(network , n_units=1000 ,
                                       act=tf.nn.relu ,
                                       name='%s_fc3_relu' % self.scope)
        return network


    def asyc_parameters(self , session):
        print "asyc_parameters"
        npz = np.load('/home/wyp/RL_toolbox/RL_classify/data/vgg16_weights.npz')
        params = []
        for val in sorted(npz.items()):
            print("  Loading %s" % str(val[0]))
            params.append(val[1])
        tl.files.assign_params(session , params , self.net)

    def get_feature(self, image_list):
        assert image_list[0].shape == (224, 224, 3)
        return self.session.run(self.feature_vector, feed_dict={self.obs:image_list})