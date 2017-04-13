from RLToolbox.algorithm.ActorCritic import ActorCritic
import tensorflow as tf
import numpy as np
from RLToolbox.toolbox.common.utils import *
from RLToolbox.toolbox.logger.logger import Logger

class ACAgent(ActorCritic):
    def __init__(self, env, session, baseline, storage, distribution, net, pms):
        super(ACAgent, self).__init__(env, session, baseline, storage, distribution, net, pms)
        self.pms = pms
        self.init_network()
        self.saver = tf.train.Saver(max_to_keep=10)

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
        if self.pms.min_std is not None:
            log_std_var = tf.maximum(self.net.action_dist_logstds_n , np.log(self.pms.min_std))
        if self.pms.max_std is not None:
            log_std_var = tf.minimum(self.net.action_dist_logstds_n , np.log(self.pms.max_std))
        self.action_dist_stds_n = tf.exp(log_std_var)
        self.old_dist_info_vars = dict(mean=self.net.old_dist_means_n , log_std=self.net.old_dist_logstds_n)
        self.new_dist_info_vars = dict(mean=self.net.action_dist_means_n , log_std=self.net.action_dist_logstds_n)
        self.likehood_action_dist = self.distribution.log_likelihood_sym(self.net.action_n , self.new_dist_info_vars)
        self.ratio_n = self.distribution.likelihood_ratio_sym(self.net.action_n , self.new_dist_info_vars ,
                                                              self.old_dist_info_vars)
        surr = - tf.reduce_mean(self.likehood_action_dist * self.net.advant)  # Surrogate loss
        self.losses = [surr]
        var_list = self.net.var_list
        self.gf = GetFlat(var_list)  # get theta from var_list
        self.gf.session = self.session
        self.sff = SetFromFlat(var_list)  # set theta from var_List
        self.sff.session = self.session
        # get g
        self.pg = flatgrad(surr , var_list)
        # get A
        self.flat_tangent = tf.placeholder(dtype , shape=[None])
        shapes = map(var_shape , var_list)
        start = 0
        tangents = []
        for shape in shapes:
            size = np.prod(shape)
            param = tf.reshape(self.flat_tangent[start:(start + size)] , shape)
            tangents.append(param)
            start += size
        self.session.run(tf.global_variables_initializer())
        self.net.asyc_parameters(session=self.session)

    def init_logger(self):
        head = ["rewards" , "std"]
        self.logger = Logger(head)

    def learn(self):
        # self.load_model(None)
        self.init_logger()
        iter_num = 0
        while True:
            print "\n********** Iteration %i ************" % iter_num
            stats , theta , thprev = self.train_mini_batch(linear_search=False)
            self.sff(theta)
            self.logger.log_row([stats["Average sum of rewards per episode"] ,
                                 self.session.run(self.net.action_dist_logstd_param)[0][0]])
            for k , v in stats.iteritems():
                print(k + ": " + " " * (40 - len(k)) + str(v))
            if iter_num % self.pms.save_model_times == 0:
                self.save_model(self.pms.environment_name + "-" + str(iter_num))
            iter_num += 1

