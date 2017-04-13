from RL_classify.pano_sence_analysis.enviroment import Enviroment
import tensorflow as tf
import tensorlayer as tl
import cv2
import numpy as np

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
        self.session = session
        self.init_classify_network()

    def init_classify_network(self):
        self.x = tf.placeholder(dtype=tf.float32 , shape=[None , 100 , 100 , 3] , name='x')
        self.y_ = tf.placeholder(dtype=tf.int64 , shape=[None , ] , name='y_')

        network = tl.layers.InputLayer(self.x , name='input_layer')
        """ conv1 """
        network = tl.layers.Conv2dLayer(network ,
                                        act=tf.nn.relu,
                                        shape=[3 , 3 , 3 , 64] ,  # 64 features for each 3x3 patch
                                        strides=[1 , 1 , 1 , 1] ,
                                        padding='SAME' ,
                                        name='conv1_1')
        network = tl.layers.Conv2dLayer(network ,
                                        act=tf.nn.relu ,
                                        shape=[3 , 3 , 64 , 64] ,  # 64 features for each 3x3 patch
                                        strides=[1 , 1 , 1 , 1] ,
                                        padding='SAME' ,
                                        name='conv1_2')
        network = tl.layers.PoolLayer(network ,
                                      ksize=[1 , 2 , 2 , 1] ,
                                      strides=[1 , 2 , 2 , 1] ,
                                      padding='SAME' ,
                                      pool=tf.nn.max_pool ,
                                      name='pool1')
        network = tl.layers.FlattenLayer(network , name='flatten')
        network = tl.layers.DenseLayer(network , n_units=800 ,
                                       act=tf.nn.relu , name='relu1')
        # network = tl.layers.DropoutLayer(network , keep=0.5 , name='drop2')
        network = tl.layers.DenseLayer(network , n_units=800 ,
                                       act=tf.nn.relu , name='relu2')
        # network = tl.layers.DropoutLayer(network , keep=0.5 , name='drop3')
        self.network = tl.layers.DenseLayer(network , n_units=self.pms.class_number ,
                                       act=tf.identity ,
                                       name='output_layer')

        self.y = y = self.network.outputs
        self.cost = tl.cost.cross_entropy(y , self.y_)
        correct_prediction = tf.equal(tf.argmax(y , 1) , self.y_)
        self.acc = tf.reduce_mean(tf.cast(correct_prediction , tf.float32))
        self.y_softmax_op = tf.nn.softmax(y)
        self.y_op = tf.argmax(self.y_softmax_op , 1)
        self.y_softmax = tf.reduce_max(self.y_softmax_op , 1)

        train_params = network.all_params
        self.train_op = tf.train.AdamOptimizer(learning_rate=0.0001 , beta1=0.9 , beta2=0.999 ,
                                          epsilon=1e-08 , use_locking=False).minimize(self.cost , var_list=train_params)

        self.session.run(tf.global_variables_initializer())

    def classify_path_image(self):
        # classify self.path_view_container
        X_train = [self.current_view]
        probabily = self.session.run(self.y_softmax_op, feed_dict={self.x:X_train})[0]
        real_label_probabily = probabily[self.current_label]
        return [real_label_probabily]

    def get_reward(self):
        probabily = 1.0
        for prob in self.path_probabily_container:
            probabily = probabily * prob[0]
        return -1.0 + probabily

    def reset(self, **kwargs):
        # get initial view
        self.env = Enviroment("/home/wyp/RL_toolbox/RL_classify/data/pano_dataset/train/train.txt")
        self.current_view, self.current_label = self.env.generate_new_scence()
        self.path_view_container = []
        self.path_label_container = []
        self.path_probabily_container = []
        self.path_view_container.append(self.current_view)
        self.path_label_container.append(self.current_label)
        probabily = self.classify_path_image()
        self.path_probabily_container.append(probabily)
        if self.episode % self.pms.train_classify_frequency == 0 and self.episode != 0:
            print "start train classify net..."
            self.train_classify()
        self.episode += 1
        return self.current_view

    def step(self, action):
        self.step_view, self.step_label = self.env.action(action)
        self.path_view_container.append(self.step_view)
        self.path_label_container.append(self.step_label)
        reward = self.get_reward()
        self.train_data_view.append(self.path_view_container)
        self.train_data_label.append(self.path_label_container)
        return self.step_view, reward, False, {"label":self.current_label}

    def train_classify(self):
        # self.train_data_view contains all train image
        # self.train_data_label contains all train label
        X_train = np.concatenate(self.train_data_view).astype(np.float32)
        y_train = np.concatenate(self.train_data_label)
        self.session.run(self.train_op, feed_dict={self.x: X_train, self.y_:y_train})
        self.train_data_label = []
        self.train_data_view = []


    def render(self, mode="human"):
        pass

