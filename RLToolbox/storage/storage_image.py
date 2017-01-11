import cv2
import numpy as np
from RLToolbox.toolbox.common.utils import *

class Storage(object):
    def __init__(self, agent, env, baseline, pms):
        self.paths = []
        self.env = env
        self.agent = agent
        self.obs = []
        self.obs_origin = []
        self.baseline = baseline
        self.pms = pms

    def get_single_path(self):
        self.obs_origin, self.obs, obs_x, actions, rewards, action_dists = [], [], [], [], [], []
        ob_x = self.env.reset()
        ob = self.env.render('rgb_array')
        # self.agent.prev_action *= 0.0
        # self.agent.prev_obs *= 0.0
        episode_steps = 0
        for _ in xrange(self.pms.max_path_length):
            self.obs_origin.append(ob)
            deal_ob = self.deal_image(ob)
            action, action_dist = self.agent.get_action(deal_ob)
            self.obs.append(deal_ob)
            obs_x.append(ob_x)
            actions.append(action)
            action_dists.append(action_dist)
            res = self.env.step(action) # res
            if self.pms.render:
                self.env.render()
            ob_x = res[0]
            ob = self.env.render('rgb_array')
            rewards.append([res[1]])
            episode_steps += 1
            if res[2]:
                break
        path = dict(
            observations=np.concatenate([self.obs]),
            obs_x=np.concatenate([obs_x]),
            agent_infos=np.concatenate([action_dists]),
            rewards=np.array(rewards),
            actions=np.array(actions),
            episode_steps=episode_steps
        )
        self.paths.append(path)
        # self.agent.prev_action *= 0.0
        # self.agent.prev_obs *= 0.0
        return path

    def get_paths(self):
        paths = self.paths
        self.paths = []
        return paths

    def process_paths(self, paths):
        sum_episode_steps = 0
        for path in paths:
            sum_episode_steps += path['episode_steps']
            path['baselines'] = self.baseline.predict(path)
            path["returns"] = np.concatenate(discount(path["rewards"] , self.pms.discount))
            path["advantages"] = path['returns'] - path['baselines']
        action_dist_n = np.concatenate([path["agent_infos"] for path in paths])
        obs_n = np.concatenate([path["observations"] for path in paths])
        obs_x_n = np.concatenate([path["obs_x"] for path in paths])
        action_n = np.concatenate([path["actions"] for path in paths])
        rewards = np.concatenate([path["rewards"] for path in paths])
        advantages = np.concatenate([path["advantages"] for path in paths])

        if self.pms.center_adv:
            advantages = (advantages - np.mean(advantages)) / (advantages.std() + 1e-8)

        self.baseline.fit(paths)

        samples_data = dict(
            observations=obs_n,
            obs_x_n=obs_x_n,
            actions=action_n,
            rewards=rewards,
            advantages=advantages,
            agent_infos=action_dist_n,
            paths=paths,
            sum_episode_steps=sum_episode_steps
        )
        return samples_data

    def deal_image(self , image):
        obs = cv2.resize(image , (self.pms.obs_height , self.pms.obs_width))
        # obs = np.transpose(np.array(obs), (2, 0, 1))
        return obs