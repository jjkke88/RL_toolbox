import multiprocessing
from RLToolbox.toolbox.common.utils import *
import gym
import time
from random import randint
import cv2


class Actor(multiprocessing.Process):
    def __init__(self, args, task_q, result_q, actor_id, monitor, pms, net_class):
        multiprocessing.Process.__init__(self)
        self.actor_id = actor_id
        self.task_q = task_q
        self.result_q = result_q
        self.args = args
        self.monitor = monitor
        self.pms = pms
        self.net_class = net_class
        # pms.max_path_length = gym.spec(args.environment_name).timestep_limit

    def get_action(self, obs):
        if self.net == None:
            raise NameError("network have not been defined")
        obs = np.expand_dims(obs , 0)
        # action_dist_logstd = np.expand_dims([np.log(pms.std)], 0)
        action_dist_means_n , action_dist_logstds_n = self.session.run(
            [self.net.action_dist_means_n, self.net.action_dist_logstds_n], feed_dict={self.net.obs: obs})
        if self.pms.train_flag:
            rnd = np.random.normal(size=action_dist_means_n[0].shape)
            action = rnd * np.exp(action_dist_logstds_n[0]) + action_dist_means_n[0]
        else:
            action = action_dist_means_n[0]
        # action = np.clip(action, pms.min_a, pms.max_a)
        return action, dict(mean=action_dist_means_n[0] , log_std=np.exp(action_dist_logstds_n[0]))

    def run(self):
        self.env = gym.make(self.args.environment_name)
        self.env.seed(randint(0, 999999))
        if self.monitor:
            self.env.monitor.start('monitor/', force=True)

        self.net = self.net_class("rollout_network" + str(self.actor_id))
        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
        self.session = tf.Session(config=config)
        var_list = self.net.var_list
        self.session.run(tf.initialize_all_variables())
        self.set_policy = SetFromFlat(var_list)
        self.set_policy.session = self.session
        while True:
            # get a task, or wait until it gets one
            next_task = self.task_q.get(block=True)
            if type(next_task) is int and next_task == 1:
                # the task is an actor request to collect experience
                path = self.rollout()
                # print "single rollout time:"+str(end-start)
                self.task_q.task_done()
                self.result_q.put(path)
            elif type(next_task) is int and next_task == 2:
                print "kill message"
                if self.monitor:
                    self.env.monitor.close()
                self.task_q.task_done()
                break
            else:
                # the task is to set parameters of the actor policy
                next_task = np.array(next_task)
                self.set_policy(next_task)
                # super hacky method to make sure when we fill the queue with set parameter tasks,
                # an actor doesn't finish updating before the other actors can accept their own tasks.
                time.sleep(0.1)
                self.task_q.task_done()
        return

    def rollout(self):
        """
        :param:observations:obs list
        :param:actions:action list
        :param:rewards:reward list
        :param:agent_infos: mean+log_std dictlist
        :param:env_infos: no use, just information about environment
        :return: a path, list
        """
        # if pms.record_movie:
        #     outdir = 'log/trpo'
        #     self.env.monitor.start(outdir , force=True)
        observations = []
        actions = []
        rewards = []
        agent_infos = []
        env_infos = []
        if self.pms.render:
            self.env.render()
        o = self.env.reset()

        episode_steps = 0
        for i in xrange(self.pms.max_path_length - 1):
            o = self.env.render('rgb_array')
            o = self.deal_image(o)
            a, agent_info = self.get_action(o)
            next_o, reward, terminal, env_info = self.env.step(a)
            observations.append(o)
            rewards.append(np.array([reward]))
            actions.append(a)
            agent_infos.append([agent_info])
            env_infos.append([])
            episode_steps += 1
            if terminal:
                break
            o = next_o
            if self.pms.render:
                self.env.render()
        path = dict(
            observations=np.array(observations) ,
            actions=np.array(actions) ,
            rewards=np.array(rewards) ,
            agent_infos=np.concatenate(agent_infos) ,
            env_infos=np.concatenate(env_infos) ,
            episode_steps=episode_steps
        )
        return path

    def deal_image(self , image):
        # index = len(self.obs_origin)
        # image_end = []
        # if index < pms.history_number:
        #     image_end = self.obs_origin[0:index]
        #     for i in range(pms.history_number - index):
        #         image_end.append(image)
        # else:
        #     image_end = self.obs_origin[index - pms.history_number:index]
        #
        # image_end = np.concatenate(image_end)
        # # image_end = image_end.reshape((pms.obs_height, pms.obs_width, pms.history_number))
        # obs = cv2.resize(cv2.cvtColor(image_end , cv2.COLOR_RGB2GRAY) / 255. , (pms.obs_height , pms.obs_width))
        obs = cv2.resize(image, (self.pms.obs_height, self.pms.obs_width))
        # obs = np.transpose(np.array(obs), (2, 0, 1))
        return obs

class ParallelStorageImage():
    def __init__(self, pms, net_class):
        self.args = pms
        self.tasks = multiprocessing.JoinableQueue()
        self.results = multiprocessing.Queue()
        self.actors = []
        self.actors.append(Actor(self.args, self.tasks, self.results, 9999, self.args.record_movie, pms, net_class))
        for i in xrange(self.args.jobs-1):
            self.actors.append(Actor(self.args, self.tasks, self.results, 37*(i+3), False, pms, net_class))
        for a in self.actors:
            a.start()
        # we will start by running 20,000 / 1000 = 20 episodes for the first ieration
        self.average_timesteps_in_episode = 1000

    def get_paths(self):
        # keep 20,000 timesteps per update
        num_rollouts = self.args.paths_number
        # print "rollout_number:"+str(num_rollouts)
        for i in xrange(num_rollouts):
            self.tasks.put(1)
        start = time.time()
        self.tasks.join()
        end = time.time()
        # print "rollout real time"+str(end-start)
        paths = []
        while num_rollouts:
            num_rollouts -= 1
            paths.append(self.results.get())
        return paths

    def set_policy_weights(self, parameters):
        for i in xrange(self.args.jobs):
            self.tasks.put(parameters)
        self.tasks.join()

    def end(self):
        for i in xrange(self.args.jobs):
            self.tasks.put(2)