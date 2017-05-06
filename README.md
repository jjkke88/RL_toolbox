# RL_toolbox
## implemented algorithm: trpo a3c
### a3c:for continous action space, use multi processes, but saving model has not been implemented.
### trpo:for continous and discrete action space
## contain some useful reinforcement learning algorithm and relative tool

# to add algorithm
* A new algorithm class should be defined like `algorithm(self, env, session, baseline, storage, distribution, net, pms)`
* A new agent class should be defined like `Agent(self, env, session, baseline, storage, distribution, net, pms)`

# storage
* to rollout and process rollout data
* A new storage class should implement three funtions 
** get single path
** get_paths
** process_paths
* A new storage should have `agent, env, baseline, pms`

# agent
* get_action(ob): action, agent_info = agent.get_action(ob)
