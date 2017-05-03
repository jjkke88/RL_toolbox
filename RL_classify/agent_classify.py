from RLToolbox.agent.TRPO_agent import TRPOAgent
import os
import cv2
import numpy as np
from RLToolbox.toolbox.math import krylov
from RLToolbox.toolbox.common.utils import *
import math
import tensorlayer as tl

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

            g = self.session.run(self.pg , feed_dict=feed)
            stepdir = krylov.cg(fisher_vector_product , -g , cg_iters=self.pms.cg_iters)
            shs = 0.5 * stepdir.dot(fisher_vector_product(stepdir))  # theta
            # if shs<0, then the nan error would appear
            if shs >= 0:
                lm = np.sqrt(shs / self.pms.max_kl)
                fullstep = stepdir / lm
                neggdotstepdir = -g.dot(stepdir)

                def loss(th):
                    self.sff(th)
                    return self.session.run(self.losses , feed_dict=feed)

                if linear_search:
                    theta_t = linesearch(loss , thprev , fullstep , neggdotstepdir / lm , max_kl=self.pms.max_kl)
                else:
                    theta_t = thprev + fullstep
                fullstep_all.append(theta_t - thprev)
        theta = thprev + np.array(fullstep_all).mean(axis=0)
        stats = {}
        stats["sum steps of episodes"] = sample_data_source["sum_episode_steps"]
        stats["Average sum of rewards per episode"] = episoderewards.mean()
        stats["surrent loss"] = loss(theta)[0]
        return stats , theta , thprev

    def test(self, model_name):
        self.load_model(model_name)
        all_image_container = []
        all_label_container = []
        for i in xrange(50):
            current_image = self.env.reset()
            current_label = self.env.current_label
            image_container = []
            label_container = []
            label_container.append(current_label)
            image_container.append(current_image)
            for j in xrange(self.pms.max_path_length):
                o, info = self.get_action(current_image)
                next_image, _, _, _ = self.env.step(o)
                image_container.append(next_image)
            all_image_container.append(image_container)
            all_label_container.append(label_container)
        train_view_length = len(all_image_container)
        t = np.concatenate(np.concatenate(all_image_container))
        X_test = np.reshape(t , (train_view_length , 100 , 100 , 3)).astype(np.float32)
        y_test = np.concatenate(all_label_container)
        tl.utils.test(self.session , self.env.network, self.env.acc, X_test, y_test, self.env.x, self.env.y_, batch_size=None, cost=self.env.cost)


