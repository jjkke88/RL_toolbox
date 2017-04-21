from RL_classify.pano_sence_analysis.enviroment import Enviroment
import tensorflow as tf
import tensorlayer as tl
import cv2
import numpy as np
from RLToolbox.toolbox.math.statistics import *
from RL_classify.VggNet import VggNet

class EnvironmentClassify():
    def __init__(self, env, pms, type="origin", session=None):
        self.path_view_container = []
        self.path_label_container = []
        self.path_length = 0
        self.episode = 0
        self.pms = pms
        self.agent = None
        self.train_data_view = []
        self.train_data_label = []
        self.train_data_feature = []
        self.session = session
        self.env = Enviroment(pms.train_file)
        self.init_classify_network()
        self.init_feature_extract_net()

    def init_classify_network(self):
        self.x = tf.placeholder(dtype=tf.float32 , shape=[None , 1000] , name='x')
        self.y_ = tf.placeholder(dtype=tf.int64 , shape=[None , ] , name='y_')

        # vgg_classify_net = VggNet(self.session, self.pms, scope="classify")
        # self.classify_network = vgg_classify_net.net
        # self.classify_network = tl.layers.DenseLayer(self.classify_network , n_units=128 ,
        #                                act=tf.nn.relu ,
        #                                name='%s_fc_train4_relu' % self.scope)
        # network = tl.layers.DenseLayer(self.classify_network , n_units=self.pms.class_number,
        #                                act=tf.nn.softmax,
        #                                name='%s_fc_train5_relu' % self.scope)
        self.classify_net = tl.layers.InputLayer(self.x, name="classify_net_input")
        self.classify_net = tl.layers.DenseLayer(self.classify_net , n_units=128 ,
                                       act=tf.nn.relu ,
                                       name='classify_fc_train4_relu')
        self.classify_net = tl.layers.DenseLayer(self.classify_net , n_units=self.pms.class_number,
                                       act=tf.nn.softmax,
                                       name='classify_fc_train5_relu')
        self.y = y = self.classify_net.outputs
        self.cost = tl.cost.cross_entropy(y , self.y_)
        correct_prediction = tf.equal(tf.argmax(y , 1) , self.y_)
        self.acc = tf.reduce_mean(tf.cast(correct_prediction , tf.float32))
        self.y_softmax_op = tf.nn.softmax(y)
        self.y_op = tf.argmax(self.y_softmax_op , 1)
        self.y_softmax = tf.reduce_max(self.y_softmax_op , 1)

        train_params = self.classify_net.all_params
        self.train_op = tf.train.AdamOptimizer(learning_rate=0.0001 , beta1=0.9 , beta2=0.999 ,
                                          epsilon=1e-08 , use_locking=False).minimize(self.cost , var_list=train_params)

        self.session.run(tf.global_variables_initializer())

    def init_feature_extract_net(self):
        self.feature_extract_net = VggNet(self.session, self.pms)

    def classify_path_image(self):
        # classify self.path_view_container
        X_train = [self.current_feature]
        probabily = self.session.run(self.y_softmax_op, feed_dict={self.x:X_train})[0]
        return probabily

    def classify_path_image_for_test(self, feature):
        # classify self.path_view_container
        X_train = [feature]
        prop = self.session.run(self.y_softmax_op , feed_dict={self.x: X_train})[0]
        return prop

    def get_reward(self):
        result = np.ones(self.path_probabily_container[0].shape)
        result = min_max_norm(result)
        for prob_list in self.path_probabily_container:
            prob_list = np.array(prob_list)
            result = min_max_norm(result * prob_list)
        return -1 + result[self.current_label]

    def reset(self, train_classify_net=True, **kwargs):
        # get initial view
        self.current_view, self.current_label = self.env.generate_new_scence()
        self.path_view_container = []
        self.path_label_container = []
        self.path_feature_container = []
        self.path_probabily_container = []
        self.current_feature = self.feature_extract_net.get_feature([self.current_view])[0]
        self.path_view_container.append(self.current_view)
        self.path_label_container.append(self.current_label)
        self.path_feature_container.append(self.current_feature)
        probabily = self.classify_path_image()
        self.path_probabily_container.append(probabily)
        if self.episode % self.pms.train_classify_frequency == 0 and self.episode != 0 and train_classify_net:
            print "start train classify net..." + str((self.episode))
            self.train_classify()
        self.episode += 1
        return self.current_feature

    def step(self, action):
        self.step_view, self.step_label = self.env.action(action)
        self.current_view = self.step_view
        self.current_feature = self.feature_extract_net.get_feature([self.current_view])[0]
        self.path_view_container.append(self.step_view)
        self.path_label_container.append(self.step_label)
        self.path_feature_container.append(self.current_feature)
        probabily = self.classify_path_image()
        self.path_probabily_container.append(probabily)
        reward = self.get_reward()
        self.train_data_view.append(self.path_view_container)
        self.train_data_label.append(self.path_label_container)
        self.train_data_feature.append(self.path_feature_container)
        return self.current_feature, reward, False, {"label":self.current_label}

    def train_classify(self):
        # self.train_data_view contains all train image
        # self.train_data_label contains all train label
        X_train = np.concatenate(self.train_data_feature).astype(np.float32)
        y_train = np.concatenate(self.train_data_label)
        self.session.run(self.train_op, feed_dict={self.x: X_train, self.y_:y_train})
        self.train_data_label = []
        self.train_data_view = []
        self.train_data_feature = []
        if self.episode % self.pms.test_frequency == 0 and self.episode != 0:
            self.agent.test(None, load=True, test_number=50)


    def render(self, mode="human"):
        pass

