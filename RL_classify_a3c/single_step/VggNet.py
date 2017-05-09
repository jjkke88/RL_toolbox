import tensorlayer as tl
import tensorflow as tf
import numpy as np

class VggNet:
    def __init__(self, session, pms, scope="VGG"):
        self.session = session
        self.scope = scope
        self.pms = pms
        with tf.variable_scope("%s_shared" % scope):
            self.x = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name='x')
            self.y_ = tf.placeholder(dtype=tf.int64, shape=[None, ], name='y_')
            network = tl.layers.InputLayer(self.x , name='%s_input_layer' % scope)
            net_cnn = self.conv_layers(network)
            network = tl.layers.FlattenLayer(net_cnn , name="%s_flatten_layer" % scope)
            self.net = self.fc_layers(network)
            self.feature_vector = self.feature.outputs
            self.y = y = self.net.outputs
            self.cost = tl.cost.cross_entropy(y, self.y_, name="classify_cost")
            correct_prediction = tf.equal(tf.argmax(y, 1), self.y_)
            self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self.y_softmax_op = tf.nn.softmax(y)
            self.y_op = tf.argmax(self.y_softmax_op, 1)
            self.y_softmax = tf.reduce_max(self.y_softmax_op, 1)
            self.asyc_parameters(self.session)

    def conv_layers(self, net_in):
        with tf.name_scope('preprocess') as scope:
            """
            Notice that we include a preprocessing layer that takes the RGB image
            with pixels values in the range of 0-255 and subtracts the mean image
            values (calculated over the entire ImageNet training set).
            """
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            net_in.outputs = net_in.outputs - mean
        """ conv1 """
        network = tl.layers.Conv2dLayer(net_in,
                                        act=tf.nn.relu,
                                        shape=[3, 3, 3, 64],  # 64 features for each 3x3 patch
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='conv1_1')
        network = tl.layers.Conv2dLayer(network,
                                        act=tf.nn.relu,
                                        shape=[3, 3, 64, 64],  # 64 features for each 3x3 patch
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='conv1_2')
        network = tl.layers.PoolLayer(network,
                                      ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1],
                                      padding='SAME',
                                      pool=tf.nn.max_pool,
                                      name='pool1')
        """ conv2 """
        network = tl.layers.Conv2dLayer(network,
                                        act=tf.nn.relu,
                                        shape=[3, 3, 64, 128],  # 128 features for each 3x3 patch
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='conv2_1')
        network = tl.layers.Conv2dLayer(network,
                                        act=tf.nn.relu,
                                        shape=[3, 3, 128, 128],  # 128 features for each 3x3 patch
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='conv2_2')
        network = tl.layers.PoolLayer(network,
                                      ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1],
                                      padding='SAME',
                                      pool=tf.nn.max_pool,
                                      name='pool2')
        """ conv3 """
        network = tl.layers.Conv2dLayer(network,
                                        act=tf.nn.relu,
                                        shape=[3, 3, 128, 256],  # 256 features for each 3x3 patch
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='conv3_1')
        network = tl.layers.Conv2dLayer(network,
                                        act=tf.nn.relu,
                                        shape=[3, 3, 256, 256],  # 256 features for each 3x3 patch
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='conv3_2')
        network = tl.layers.Conv2dLayer(network,
                                        act=tf.nn.relu,
                                        shape=[3, 3, 256, 256],  # 256 features for each 3x3 patch
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='conv3_3')
        network = tl.layers.PoolLayer(network,
                                      ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1],
                                      padding='SAME',
                                      pool=tf.nn.max_pool,
                                      name='pool3')
        """ conv4 """
        network = tl.layers.Conv2dLayer(network,
                                        act=tf.nn.relu,
                                        shape=[3, 3, 256, 512],  # 512 features for each 3x3 patch
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='conv4_1')
        network = tl.layers.Conv2dLayer(network,
                                        act=tf.nn.relu,
                                        shape=[3, 3, 512, 512],  # 512 features for each 3x3 patch
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='conv4_2')
        network = tl.layers.Conv2dLayer(network,
                                        act=tf.nn.relu,
                                        shape=[3, 3, 512, 512],  # 512 features for each 3x3 patch
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='conv4_3')
        network = tl.layers.PoolLayer(network,
                                      ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1],
                                      padding='SAME',
                                      pool=tf.nn.max_pool,
                                      name='pool4')
        """ conv5 """
        network = tl.layers.Conv2dLayer(network,
                                        act=tf.nn.relu,
                                        shape=[3, 3, 512, 512],  # 512 features for each 3x3 patch
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='conv5_1')
        network = tl.layers.Conv2dLayer(network,
                                        act=tf.nn.relu,
                                        shape=[3, 3, 512, 512],  # 512 features for each 3x3 patch
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='conv5_2')
        network = tl.layers.Conv2dLayer(network,
                                        act=tf.nn.relu,
                                        shape=[3, 3, 512, 512],  # 512 features for each 3x3 patch
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='conv5_3')
        network = tl.layers.PoolLayer(network,
                                      ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1],
                                      padding='SAME',
                                      pool=tf.nn.max_pool,
                                      name='pool5')
        return network

    def fc_layers(self, net):
        network = tl.layers.FlattenLayer(net, name='flatten')
        network = tl.layers.DenseLayer(network, n_units=4096,
                                       act=tf.nn.relu,
                                       name='fc1_relu')
        self.feature = network = tl.layers.DenseLayer(network, n_units=4096,
                                       act=tf.nn.relu,
                                       name='fc2_relu')
        network = tl.layers.DenseLayer(network, n_units=26,
                                       act=tf.identity,
                                       name='fc3_relu')
        return network

    def asyc_parameters(self , session):
        # print "asyc_parameters"
        # npz = np.load('/home/aqrose/RL_toolbox/RL_classify/single_step/model.npz')
        # params = []
        # for val in sorted(npz.items()):
        #     print("  Loading %s" % str(val[0]))
        #     params.append(val[1])
        # tl.files.assign_params(session , params , self.net)

        tl.files.load_and_assign_npz(session, '/home/aqrose/RL_toolbox/RL_classify/single_step/model.npz', self.net)

    def get_feature_and_prob(self, image_list):
        assert image_list[0].shape == (224, 224, 3)
        return self.session.run([self.feature_vector, self.y_softmax_op], feed_dict={self.x:image_list})