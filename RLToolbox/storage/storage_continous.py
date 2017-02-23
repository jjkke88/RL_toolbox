import math

from RLToolbox.toolbox.common.utils import *


class Storage(object):
    def __init__(self, agent, env, baseline, pms):
        self.paths = []
        self.env = env
        self.agent = agent
        self.baseline = baseline
        self.pms = pms

    def get_single_path(self):
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
        o = self.env.reset()
        if self.pms.render:
            self.env.render()
        o = self.env.reset()
        if self.pms.obs_as_image:
            o = self.env.render('rgb_array')
        episode_steps = 0
        while episode_steps< self.pms.max_path_length:
            a, agent_info = self.agent.get_action(o)

            next_o, reward, terminal, env_info = self.env.step(a)
            if self.pms.obs_as_image:
                next_o = self.env.render('rgb_array')
            observations.append(o)
            rewards.append(np.array([reward]))
            actions.append(a)
            agent_infos.append([agent_info])
            env_infos.append([env_info])
            episode_steps += 1
            if terminal:
                break
            o = next_o
            if self.pms.render:
                self.env.render()
        self.paths.append(dict(
            observations=np.array(observations),
            actions=np.array(actions),
            rewards=np.array(rewards),
            agent_infos=np.concatenate(agent_infos),
            env_infos=np.concatenate(env_infos),
            episode_steps=episode_steps
        ))

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
        observations = np.concatenate([path["observations"] for path in paths])
        actions = np.concatenate([path["actions"] for path in paths])
        rewards = np.concatenate([path["rewards"] for path in paths])
        advantages = np.concatenate([path["advantages"] for path in paths])
        env_infos = np.concatenate([path["env_infos"] for path in paths])
        agent_infos = np.concatenate([path["agent_infos"] for path in paths])
        if self.pms.center_adv:
            advantages -= np.mean(advantages)
            advantages /= (advantages.std() + 1e-8)

        # for some unknown reaseon, it can not be used
        # if pms.positive_adv:
        #     advantages = (advantages - np.min(advantages)) + 1e-8

        # average_discounted_return = \
        #     np.mean([path["returns"][0] for path in paths])
        #
        # undiscounted_returns = [sum(path["rewards"]) for path in paths]


        # ev = self.explained_variance_1d(
        #     np.concatenate(baselines),
        #     np.concatenate(returns)
        # )
        samples_data = dict(
            observations=observations,
            actions=actions,
            rewards=rewards,
            advantages=advantages,
            env_infos=env_infos,
            agent_infos=agent_infos,
            paths=paths,
            sum_episode_steps=sum_episode_steps
        )
        self.baseline.fit(paths)
        return samples_data

    def explained_variance_1d(ypred, y):
        assert y.ndim == 1 and ypred.ndim == 1
        vary = np.var(y)
        if np.isclose(vary, 0):
            if np.var(ypred) > 0:
                return 0
            else:
                return 1
        if abs(1 - np.var(y - ypred) / (vary + 1e-8)) > 1e5:
            import ipdb;
            ipdb.set_trace()
        return 1 - np.var(y - ypred) / (vary + 1e-8)

