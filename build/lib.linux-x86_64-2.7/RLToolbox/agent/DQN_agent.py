import tensorflow as tf
import numpy as np
import math
from RLToolbox.toolbox.sample.e_greedy import EGreedy

class DQNAgent(object):
    def __init__(self, env, session, baseline, storage, distribution, net, pms):
        self.env = env
        self.session = session
        self.baseline = baseline
        self.storage = storage
        self.distribution = distribution
        self.net = net
        self.pms = pms
        self.e_greedy = 1
        self.sample_policy = EGreedy()

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
        self.net.gf_target.session = self.session
        self.session.run(tf.initialize_all_variables())
        self.update_target_net()

    def get_samples(self, path_number):
        """
        command storage to rollout
        :param path_number: path numbers
        :return: None
        """
        for i in xrange(path_number):
            steps = self.storage.get_single_path()
            print "path length:%d"%steps

    def get_action(self, obs):
        sample_type = self.sample_policy.get_sample_type(self.e_greedy)
        if sample_type == "RANDOM":
            action = self.env.action_space.sample()
        elif sample_type == "POLICY":
            obs = np.expand_dims(obs, 0)
            action = self.session.run(self.net.action_n, feed_dict={self.net.obs:obs})[0]
        return action, dict(mean=action)

    # def learn(self):
    #     for episode in xrange(self.pms.max_episode):
    #         self.e_greedy = self.pms.episode_end + max(0., (self.pms.episode_start - self.pms.episode_end) * (
    #         self.pms.max_episode - max(0., episode)) / self.pms.max_episode)
    #         if self.e_greedy < 0.2:
    #             self.e_greedy = 0
    #         self.get_samples(self.pms.paths_number)
    #         overall_size = self.storage.getBufferSize()
    #         if overall_size > self.pms.start_train_size:
    #             for index in xrange(self.pms.train_number):
    #                 iterations = episode*self.pms.train_number + index
    #                 stats, q_acted = self.train_paths(None, iterations=iterations)
    #             print "\n********** Iteration %d ************" % iterations
    #             for k , v in stats.iteritems():
    #                 print(k + ": " + " " * (40 - len(k)) + str(v))
    #             print
    #             if iterations % self.pms.save_model_times == 0:
    #                 self.save_model(self.pms.environment_name + "-" + str(iterations))
    #             if iterations % self.pms.target_update_step == 0:
    #                 self.net.update_target_net()

    def learn(self):
        all_steps = 0
        for step in range(self.pms.max_steps):
            o = self.env.reset()
            episode_steps = 0
            stats = {}
            while episode_steps < self.pms.max_path_length:
                self.e_greedy = self.pms.episode_end + max(0. , (self.pms.episode_start - self.pms.episode_end) * (
                            self.pms.max_steps - max(0., all_steps)) / self.pms.max_steps)
                a , agent_info = self.get_action(o)
                next_o , reward , terminal , env_info = self.env.step(a)
                self.storage.saveTuple(o , a , reward , terminal)
                episode_steps += 1
                all_steps += 1
                o = next_o
                if self.pms.render:
                    self.env.render()
                overall_size = self.storage.getBufferSize()
                if overall_size > self.pms.start_train_size:
                    if all_steps % self.pms.train_frequency == 0:
                        for index in xrange(self.pms.train_number):
                            stats, q_acted = self.train_paths(None, iterations=all_steps)
                if all_steps % self.pms.target_update_step == 0:
                    self.update_target_net()
                stats['path_length'] = episode_steps
                if terminal:
                    break
            print
            print "\n********** Iteration %d ************" % step
            for k , v in stats.iteritems():
                print(k + ": " + " " * (40 - len(k)) + str(v))


            self.save_model(self.pms.environment_name + "-" + str(step))

    def test(self, model_name):
        self.load_model(model_name)
        self.net.update_target_net()
        self.e_greedy = 0.0
        self.get_samples(self.pms.paths_number)

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
        _, loss, q_acted, learning_rate = self.session.run([self.net.opt, self.net.loss, self.net.q, self.net.learning_rate_op],
                                         feed_dict={self.net.target_value: np.resize(target_value, (self.pms.batch_size,))
                                             , self.net.action_acted: self.action_data
                                             , self.net.learning_rate_step: iterations
                                             , self.net.obs: self.s_t_data})
        stats = {}
        stats["Average sum of rewards per episode"] = self.reward_data.mean()
        stats["loss"] = loss
        stats['sample number'] = len(self.reward_data)
        stats['e_greedy'] = self.e_greedy
        stats['learning_rate'] = learning_rate
        return stats, q_acted

    def update_target_net(self):
        self.net.update_target_net()

    def save_model(self, model_name):
        self.saver.save(self.session, "checkpoint/" + model_name + ".ckpt")

    def load_model(self, model_name):
        try:
            if model_name is not None:
                self.saver.restore(self.session, model_name)
                print "load model %s success" % (model_name)
            else:
                self.saver.restore(self.session, tf.train.latest_checkpoint(self.pms.checkpoint_dir))
                print "load model %s success" % (tf.train.latest_checkpoint(self.pms.checkpoint_dir))
        except:
            print "load model %s fail" % (model_name)

