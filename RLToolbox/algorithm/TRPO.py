import tensorflow as tf
import numpy as np
import math
from RLToolbox.toolbox.math import krylov
from RLToolbox.toolbox.common.utils import *
"""
Base class for TRPOAgent
"""
class TRPO(object):
    def __init__(self, env, session, baseline, storage, distribution, net, pms):
        self.env = env
        self.session = session
        self.baseline = baseline
        self.storage = storage
        self.distribution = distribution
        self.net = net
        self.pms = pms

    def init_network(self):
        raise NotImplementedError

    def get_samples(self, path_number):
        """
        command storage to rollout
        :param path_number: path numbers
        :return: None
        """
        for i in range(path_number):
            self.storage.get_single_path()

    def get_action(self, obs, *args):
        """
        get action from agent
        :param obs: observations
        :param args:
        :return: action, agent result(mean, log_std)
        """
        if self.net==None:
            raise NameError("network have not been defined")
        obs = np.expand_dims(obs, 0)
        # action_dist_logstd = np.expand_dims([np.log(pms.std)], 0)
        action_dist_means_n, action_dist_stds_n = self.session.run([self.net.action_dist_means_n, self.action_dist_stds_n],
                                               {self.net.obs: obs})
        if self.pms.train_flag:
            rnd = np.random.normal(size=action_dist_means_n[0].shape)
            action = rnd * action_dist_stds_n[0] + action_dist_means_n[0]
        else:
            action = action_dist_means_n[0]
        # action = np.clip(action, pms.min_a, pms.max_a)
        # print action
        return action, dict(mean=action_dist_means_n[0], log_std=action_dist_stds_n[0])

    def train_mini_batch(self, parallel=False, linear_search=True):
        """
        train by mini batch
        :param parallel: in this version, patallel is not aviable
        :param linear_search: whether use linear search
        :return: stats , theta , thprev (status, new theta, old theta)
        """
        # Generating paths.
        self.get_samples(self.pms.paths_number)
        paths = self.storage.get_paths()  # get_paths
        # Computing returns and estimating advantage function.
        return self.train_paths(paths, parallel=parallel, linear_search=linear_search)


    def train_paths(self, paths, parallel=False, linear_search=True):
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
        return stats , theta , thprev

    def learn(self):
        raise NotImplementedError

    def test(self, model_name):
        self.load_model(model_name)
        if self.pms.record_movie:
            for i in range(100):
                self.storage.get_single_path()
            self.env.env.monitor.close()
        else:
            for i in range(50):
                self.storage.get_single_path()

    def save_model(self, model_name):
        self.saver.save(self.session, "checkpoint/" + model_name + ".ckpt")

    def load_model(self, model_name):
        try:
            if model_name is not None:
                print "load model %s success" % (model_name)
                self.saver.restore(self.session, model_name)
            else:
                print "load model %s success" % (tf.train.latest_checkpoint(self.pms.checkpoint_dir))
                self.saver.restore(self.session, tf.train.latest_checkpoint(self.pms.checkpoint_dir))
        except:
            print "load model %s fail" % (model_name)

    def save_summary(self):
        print "write summary"
        # writer = tf.train.SummaryWriter('logs/', self.session.graph)