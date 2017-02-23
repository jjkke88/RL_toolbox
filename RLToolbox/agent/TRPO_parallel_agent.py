from RLToolbox.toolbox.common.utils import *
import numpy as np
import tensorflow as tf
import multiprocessing
from RLToolbox.toolbox.math import krylov
import time
import math
from RLToolbox.toolbox.logger.logger import Logger
from RLToolbox.algorithm.TRPO import TRPO

seed = 1
np.random.seed(seed)
tf.set_random_seed(seed)


"""
class for continoust action space in multi process
"""
class TRPOParallelAgent(multiprocessing.Process):

    def __init__(self, env, session, baseline, storage, distribution, net, pms, task_q , result_q):
        multiprocessing.Process.__init__(self)
        self.task_q = task_q
        self.result_q = result_q
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.pms = pms
        self.baseline = baseline
        self.distribution = distribution
        self.storage = storage
        self.net = net
        self.init_logger()

    def init_network(self):
        """
        [input]
        self.obs
        self.action_n
        self.advant
        self.old_dist_means_n
        self.old_dist_logstds_n
        [output]
        self.action_dist_means_n
        self.action_dist_logstds_n
        var_list
        """
        config = tf.ConfigProto(
            device_count={'GPU': 0},
        )
        config.gpu_options.per_process_gpu_memory_fraction = self.pms.GPU_fraction

        self.session = tf.Session(config=config)
        if self.pms.min_std is not None:
            log_std_var = tf.maximum(self.net.action_dist_logstds_n, np.log(self.pms.min_std))
        self.action_dist_stds_n = tf.exp(log_std_var)
        self.old_dist_info_vars = dict(mean=self.net.old_dist_means_n, log_std=self.net.old_dist_logstds_n)
        self.new_dist_info_vars = dict(mean=self.net.action_dist_means_n, log_std=self.net.action_dist_logstds_n)
        self.likehood_action_dist = self.distribution.log_likelihood_sym(self.net.action_n, self.new_dist_info_vars)
        self.ratio_n = self.distribution.likelihood_ratio_sym(self.net.action_n, self.new_dist_info_vars,
                                                              self.old_dist_info_vars)
        surr = -tf.reduce_mean(self.ratio_n * self.net.advant)  # Surrogate loss
        batch_size = tf.shape(self.net.obs)[0]
        batch_size_float = tf.cast(batch_size , tf.float32)
        kl = tf.reduce_mean(self.distribution.kl_sym(self.old_dist_info_vars, self.new_dist_info_vars))
        ent = self.distribution.entropy(self.old_dist_info_vars)
        # ent = tf.reduce_sum(-p_n * tf.log(p_n + eps)) / Nf
        self.losses = [surr, kl, ent]
        var_list = self.net.var_list

        self.gf = GetFlat(var_list)  # get theta from var_list
        self.gf.session = self.session
        self.sff = SetFromFlat(var_list)  # set theta from var_List
        self.sff.session = self.session
        # get g
        self.pg = flatgrad(surr, var_list)
        # get A
        # KL divergence where first arg is fixed
        # replace old->tf.stop_gradient from previous kl
        kl_firstfixed = self.distribution.kl_sym_firstfixed(self.new_dist_info_vars) / batch_size_float
        grads = tf.gradients(kl_firstfixed, var_list)
        self.flat_tangent = tf.placeholder(dtype, shape=[None])
        shapes = map(var_shape, var_list)
        start = 0
        tangents = []
        for shape in shapes:
            size = np.prod(shape)
            param = tf.reshape(self.flat_tangent[start:(start + size)], shape)
            tangents.append(param)
            start += size
        self.gvp = [tf.reduce_sum(g * t) for (g, t) in zip(grads, tangents)]
        self.fvp = flatgrad(tf.reduce_sum(self.gvp), var_list)  # get kl''*p
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=5)
        self.net.asyc_parameters(session=self.session)

    def init_logger(self):
        head = ["factor", "rewards", "std"]
        self.logger = Logger(head)

    def run(self):
        self.init_network()
        self.trpo = TRPO(self.env , self.session , self.baseline, self.storage, self.distribution , self.net, self.pms)
        self.trpo.saver = self.saver
        while True:
            paths = self.task_q.get()
            if paths is None:
                # kill the learner
                self.task_q.task_done()
                break
            elif paths == 1:
                # just get params, no learn
                self.task_q.task_done()
                self.result_q.put(self.gf())
            elif paths[0] == 2:
                # adjusting the max KL.
                self.pms.max_kl = paths[1]
                if paths[2] == 1:
                    print "saving checkpoint..."
                    self.trpo.save_model(self.pms.environment_name + "-" + str(paths[3]))
                    self.trpo.save_summary()
                self.task_q.task_done()
            else:
                stats , theta, thprev = self.train_paths(paths, parallel=True, linear_search=False, storage=self.storage)
                self.sff(theta)
                self.task_q.task_done()
                self.result_q.put((stats, theta, thprev))
        return

    def train_paths(self , paths , parallel=False , linear_search=True, storage=None):
        """
        train with paths data
        :param paths: source paths data
        :param parallel: not used
        :param linear_search: whether linear search
        :return: stats , theta , thprev (status, new theta, old theta)
        """
        sample_data_source = storage.process_paths(paths)
        agent_infos_source = sample_data_source["agent_infos"]
        obs_n_source = sample_data_source["observations"]
        action_n_source = sample_data_source["actions"]
        advant_n_source = sample_data_source["advantages"]
        n_samples = len(obs_n_source)
        fullstep_all = []
        episoderewards = np.array([path["rewards"].sum() for path in paths])
        thprev = self.gf()  # get theta_old
        train_number = int(1.0/self.pms.subsample_factor)
        print "train paths iterations:%d, batch size:%d"%(train_number, int(n_samples * self.pms.subsample_factor))

        for iteration in xrange(train_number):
            print "train mini batch%d"%iteration
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



