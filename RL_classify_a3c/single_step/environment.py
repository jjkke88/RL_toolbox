from RL_classify_a3c.pano_sence_analysis.enviroment import Enviroment
import tensorflow as tf
import tensorlayer as tl
import cv2
import numpy as np
from RLToolbox.toolbox.math.statistics import *
from RL_classify.single_step.VggNet import VggNet
import os

class EnvironmentClassify():
    def __init__(self, env, pms, type="train", session=None):
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
        self.type = type
        self.env = Enviroment(pms.train_file)
        self.env_test = Enviroment(pms.test_file)
        self.init_feature_extract_net()
        self.epoch_count = 0

    def init_feature_extract_net(self):
        self.feature_extract_net = VggNet(self.session, self.pms)

    def get_reward(self):
        result = self.get_prob_result()
        return -2 + result[self.current_label] + np.tanh(np.array(result.std()))

    def get_prob_result(self):
        result = np.ones(self.path_probabily_container[0].shape)
        result = min_max_norm(result)
        for prob_list in self.path_probabily_container:
            prob_list = np.array(prob_list)
            result = min_max_norm(result * prob_list)
        return result

    def reset(self, train_classify_net=True, **kwargs):
        self.epoch_count += 1
        self.path_view_container = []
        self.path_label_container = []
        self.path_feature_container = []
        self.path_probabily_container = []
        # get initial view
        if self.type == "train":
            self.current_view, self.current_label = self.env.generate_new_scence()
        else:
            self.current_view, self.current_label = self.env_test.generate_new_scence()
            cv2.imwrite(
                os.path.join(self.pms.test_image_dir, str(self.epoch_count) + str(len(self.path_view_container)) + ".jpg"),
                self.current_view)

        self.current_feature, probabily = self.feature_extract_net.get_feature_and_prob([self.current_view])
        probabily = self.agent.get_softmax_probs(self.current_feature)
        self.path_view_container.append(self.current_view)
        self.path_label_container.append(self.current_label)
        self.path_feature_container.append(self.current_feature[0])
        self.path_probabily_container.append(probabily[0])
        self.current_reward = self.last_reward = self.get_reward()
        # if self.episode % self.pms.train_classify_frequency == 0 and self.episode != 0 and train_classify_net:
        #     print "start train classify net..." + str((self.episode))
        #     self.train_classify()
        self.episode += 1
        return self.current_feature[0], self.current_label

    def step(self, action):
        if self.type == "train":
            self.step_view, self.step_label = self.env.action(action)
        else:
            self.step_view, self.step_label = self.env_test.action(action)
            cv2.imwrite(os.path.join(self.pms.test_image_dir, str(self.epoch_count) + str(len(self.path_view_container)) + ".jpg"),
                        self.step_view)
        self.current_view = self.step_view
        self.current_feature, probabily = self.feature_extract_net.get_feature_and_prob([self.current_view])
        probabily = self.agent.get_softmax_probs(self.current_feature)
        self.path_view_container.append(self.step_view)
        self.path_label_container.append(self.step_label)
        self.path_feature_container.append(self.current_feature[0])
        self.path_probabily_container.append(probabily[0])
        self.current_reward = self.get_reward()
        reward = self.current_reward - self.last_reward
        self.last_reward = self.current_reward
        self.train_data_view.append(self.path_view_container)
        self.train_data_label.append(self.path_label_container)
        self.train_data_feature.append(self.path_feature_container)
        return self.current_feature[0], reward, False, {"label":self.current_label}

    def render(self, mode="human"):
        pass

