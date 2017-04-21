from RLToolbox.agent.TRPO_agent import TRPOAgent
import os
import cv2
import numpy as np
from RLToolbox.toolbox.math import krylov
from RLToolbox.toolbox.common.utils import *
import math
import tensorlayer as tl
from RL_classify.single_step.environment import EnvironmentClassify
from RL_classify.pano_sence_analysis.enviroment import Enviroment
from RLToolbox.toolbox.math.statistics import *

class ClassifyAgent(TRPOAgent):
    def __init__(self , env , session , baseline , storage , distribution , net , pms):
        super(ClassifyAgent , self).__init__(env , session , baseline , storage , distribution , net , pms)

    def train_paths(self , paths , parallel=False , linear_search=True):
        """
        train with paths data
        :param paths: source paths data
        :param parallel: not used
        :param linear_search: whether linear search
        :return: stats , theta , thprev (status, new theta, old theta)
        """

        sample_data_source = self.storage.process_paths(paths)
        agent_infos_source = sample_data_source["agent_infos"]
        obs_n_source = sample_data_source["observations"]
        action_n_source = sample_data_source["actions"]
        advant_n_source = sample_data_source["advantages"]
        n_samples = len(obs_n_source)
        fullstep_all = []
        loss_all = []
        episoderewards = np.array([path["rewards"].sum() for path in paths])
        thprev = self.gf()  # get theta_old
        train_number = int(1.0 / self.pms.subsample_factor)
        print "train paths iterations:%d, batch size:%d" % (train_number , int(n_samples * self.pms.subsample_factor))

        for iteration in xrange(train_number):
            print "train mini batch%d" % iteration
            inds = np.random.choice(n_samples , int(math.floor(n_samples * self.pms.subsample_factor)) , replace=False)
            # inds = range(n_samples)
            obs_n = obs_n_source[inds]
            action_n = action_n_source[inds]
            advant_n = advant_n_source[inds]
            action_dist_means_n = np.array([agent_info["mean"] for agent_info in agent_infos_source[inds]])
            action_dist_logstds_n = np.array([agent_info["log_std"] for agent_info in agent_infos_source[inds]])
            feed = {self.net.obs: obs_n ,
                    self.net.advant: advant_n ,
                    self.net.old_dist_means_n: action_dist_means_n ,
                    self.net.old_dist_logstds_n: action_dist_logstds_n ,
                    self.net.action_n: action_n
                    }

            def fisher_vector_product(p):
                feed[self.flat_tangent] = p
                return self.session.run(self.fvp , feed) + self.pms.cg_damping * p

            def loss(th):
                self.sff(th)
                return self.session.run(self.losses , feed_dict=feed)


            g = self.session.run(self.pg , feed_dict=feed)
            stepdir = krylov.cg(fisher_vector_product , -g , cg_iters=self.pms.cg_iters)
            shs = 0.5 * stepdir.dot(fisher_vector_product(stepdir))  # theta
            # if shs<0, then the nan error would appear
            if shs >= 0:
                loss_all.append(loss(thprev)[0])
                lm = np.sqrt(shs / self.pms.max_kl)
                fullstep = stepdir / lm
                neggdotstepdir = -g.dot(stepdir)

                if linear_search:
                    theta_t = linesearch(loss , thprev , fullstep , neggdotstepdir / lm , max_kl=self.pms.max_kl)
                else:
                    theta_t = thprev + fullstep
                fullstep_all.append(theta_t - thprev)
        theta = thprev + np.array(fullstep_all).mean(axis=0)
        stats = {}
        stats["sum steps of episodes"] = sample_data_source["sum_episode_steps"]
        stats["Average sum of rewards per episode"] = episoderewards.mean()
        stats["surrent loss"] = np.mean(np.array(loss_all))

        return stats , theta , thprev

    def get_test_result(self, all_prop_container):
        result = np.ones(all_prop_container[0].shape)
        result = min_max_norm(result)
        for prob_list in all_prop_container:
            prob_list = np.array(prob_list)
            result = min_max_norm(result * prob_list)
        return np.argmax(result)

    def test(self, model_name, load=False, test_number=50):
        if load:
            self.load_model(model_name)
        all_image_container = []
        all_label_container = []
        test_env = Enviroment(self.pms.test_file)
        classify_right_number = 0
        for i in xrange(test_number):
            all_prop_container = []
            current_view , current_label = test_env.generate_new_scence()
            current_feature = self.env.feature_extract_net.get_feature([current_view])[0]
            image_container = []
            label_container = []
            label_container.append(current_label)
            image_container.append(current_view)
            all_prop = self.env.classify_path_image_for_test(current_feature)
            all_prop_container.append(all_prop)
            for j in xrange(self.pms.max_path_length):
                action, info = self.get_action(current_feature)
                next_image, _ = test_env.action(action)
                next_feature = self.env.feature_extract_net.get_feature([next_image])[0]
                image_container.append(next_image)
                label_container.append(current_label)
                all_prop = self.env.classify_path_image_for_test(next_feature)
                all_prop_container.append(all_prop)
            result = self.get_test_result(all_prop_container)
            if result == current_label:
                classify_right_number += 1
        print "acc:%d"%(classify_right_number)



