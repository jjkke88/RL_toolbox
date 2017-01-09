import tensorflow as tf
import numpy as np
import math

class DQNAgent(object):
    def __init__(self, env, session, baseline, storage, distribution, net, pms):
        self.env = env
        self.session = session
        self.baseline = baseline
        self.storage = storage
        self.distribution = distribution
        self.net = net
        self.pms = pms

    def init_network(self):
        """
        network 1:
        input:obs
        output:q_n
        network target:
        input obs:
        output:q_n
        """
        self.net.gf.session = self.session
        self.net.sff.session = self.session
        self.session.run(tf.initialize_all_variables())
        self.update_target_net()

    def get_samples(self, path_number):
        """
        command storage to rollout
        :param path_number: path numbers
        :return: None
        """
        for i in range(path_number):
            self.storage.get_single_path()

    def get_action(self, obs):
        obs = np.expand_dims(obs, 0)
        action = self.session.run(self.net.action_n, feed_dict={self.net.obs:obs})[0]
        return action, dict(mean=action)

    def learn(self):
        iter_num = 0
        while True:
            print "\n********** Iteration %i ************" % iter_num
            loss, q_acted = self.train_mini_batch(linear_search=False, iterations=iter_num)
            print "loss:%f"%loss
            if iter_num % self.pms.save_model_times == 0:
                self.save_model(self.pms.environment_name + "-" + str(iter_num))
            iter_num += 1

    def train_mini_batch(self, parallel=False, linear_search=True, iterations=0):
        """
        train by mini batch
        :param parallel: in this version, patallel is not aviable
        :param linear_search: whether use linear search
        :return: stats , theta , thprev (status, new theta, old theta)
        """
        # Generating paths.
        self.get_samples(self.pms.paths_number)
        paths = None  # get_paths
        # Computing returns and estimating advantage function.
        return self.train_paths(paths, parallel=parallel, linear_search=linear_search, iterations=iterations)

    def train_paths(self, paths, parallel=False, linear_search=True, iterations=0):
        """
        train with paths data
        :param paths: source paths data
        :param parallel: not used
        :param linear_search: whether linear search
        :return: stats , theta , thprev (status, new theta, old theta)
        """
        sample_data = self.storage.process_paths(paths)
        [self.s_t_data, self.action_data, self.s_t_plus_1_data, self.done_data, self.reward_data] = sample_data
        # get target q value
        targetQ = self.session.run(self.net.target_q, {self.net.obs: self.s_t_plus_1_data})
        maxQ = np.max(targetQ, axis=1)
        maxQ = np.resize(maxQ, (self.pms.batch_size, 1))
        target_value = self.done_data * maxQ * self.pms.discount + self.reward_data
        # train
        _, loss, q_acted = self.session.run([self.net.opt, self.net.loss, self.net.q],
                                         feed_dict={self.net.target_value: np.resize(target_value, (self.pms.batch_size,))
                                             , self.net.action_acted: self.action_data
                                             , self.net.learning_rate_step: iterations
                                             , self.net.obs: self.s_t_data})
        return loss, q_acted

    def update_target_net(self):
        self.net.update_target_net()

    def save_model(self, model_name):
        self.saver.save(self.session, "checkpoint/" + model_name + ".ckpt")

    def load_model(self, model_name):
        try:
            if model_name is not None:
                self.saver.restore(self.session, model_name)
            else:
                self.saver.restore(self.session, tf.train.latest_checkpoint(self.pms.checkpoint_dir))
        except:
            print "load model %s fail" % (model_name)

