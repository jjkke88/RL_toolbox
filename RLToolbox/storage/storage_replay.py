#coding=utf-8
import numpy as np
import random

class StorageReplay(object):
    def __init__(self, agent, env, baseline, pms):
        self.size = 0
        self.max_size = int(pms.buffer_size)
        self.isFull = False
        self.pms = pms
        self.env = env
        self.agent = agent
        self.state = np.zeros((pms.buffer_size, pms.obs_shape))
        self.action = np.zeros((pms.buffer_size, pms.action_shape))
        self.reward = np.zeros((pms.buffer_size, 1))
        self.done = np.zeros((pms.buffer_size, 1))
        self.s_t_data = np.zeros([pms.batch_size, pms.obs_shape], dtype="float32")
        self.s_t_plus_1_data = np.zeros([pms.batch_size, pms.obs_shape], dtype="float32")
        self.action_data = np.zeros([pms.batch_size, pms.action_shape], dtype="float32")
        self.reward_data = np.zeros([pms.batch_size, 1], dtype="float32")
        self.done_data = np.zeros([pms.batch_size, 1], dtype="float32")

    def process_paths(self, paths):
        all_selected_items = []
        for index in xrange(0, self.pms.batch_size):
            while (True):
                selected = random.randint(0, self.getBufferSize() - 1)
                # if(self.done[selected]==0):# 应该可以使用done掉的数据，因为计算target时，done掉的数据其实与下一帧无关
                #     continue
                if (selected == self.getSize() - 1):  # 减1是因为self.size一直指向下一个可写位置
                    continue
                if selected in all_selected_items:
                    continue
                break
            all_selected_items.append(selected)
            self.s_t_plus_1_data[index] = self.state[(selected + 1) % self.getBufferSize()]
            self.s_t_data[index] = self.state[selected]
            self.action_data[index] = self.action[selected]
            self.reward_data[index] = self.reward[selected]
            self.done_data[index] = self.done[selected]
        return [self.s_t_data, self.action_data, self.s_t_plus_1_data, self.done_data, self.reward_data]

    def get_single_path(self):
        """
        :param:observations:obs list
        :param:actions:action list
        :param:rewards:reward list
        :param:agent_infos: mean+log_std dictlist
        :param:env_infos: no use, just information about environment
        :return: a path, list
        """
        o = self.env.reset()
        episode_steps = 0
        while episode_steps < self.pms.max_path_length:
            a, agent_info = self.agent.get_action(o)
            next_o, reward, terminal, env_info = self.env.step(a)
            self.saveTuple(o, a, reward, terminal)
            episode_steps += 1
            if terminal:
                break
            o = next_o
            if self.pms.render:
                self.env.render()
        return episode_steps

    def saveTuple(self, state, action_num, reward, done):
        action = np.zeros(self.pms.action_shape)
        action[action_num] = 1
        if self.size < self.max_size:
            self.state[self.size] = state
            self.action[self.size] = action
            self.reward[self.size] = reward
            self.done[self.size] = 0 if done else 1
            self.size += 1
        else:
            self.state[0] = state
            self.action[0] = action
            self.reward[0] = reward
            self.done[0] = 0 if done else 1
            self.isFull = True
            self.size = 1

    def getBufferSize(self):  # 得到的是buffer被占据的情况
        if self.isFull:
            return self.max_size
        else:
            return self.size

    def getSize(self):  # 得到的是当前的指针
        return self.size

    def getIsFull(self):
        return self.isFull

    def getCurActionHistory(self):
        return self.curActionHistory

