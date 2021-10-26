# Support for Swimmer-v2, HalfCheetah-v2, Hopper-v2. Walker2d-v2, Ant-v2
import numpy as np
import torch

from gym import Env, make

from envs.multiagent.environment import MultiAgentEnv
import envs.multiagent.scenarios as scenarios

class MultiEnv(Env):

    def __init__(self, env_id, n_envs, seed, gym_env = True):

        if gym_env:
            self.envs = [make(env_id) for i in range(n_envs)]
            for env in self.envs:
                env.seed(seed)
            self.action_shape = self.envs[0].action_space.shape[0]
            self.obs_shape = self.envs[0].observation_space.shape[0]
            self.n_agents = 1
        else:
            self.envs = [MultiAgentEnvWrapper(env_id, seed)]

        self.gym_env = gym_env
        self.n_envs = n_envs

    def step(self, actions):

        obs = np.zeros((self.n_envs, self.n_agents, self.obs_shape))
        reward = np.zeros((self.n_envs, self.n_agents, 1))
        done = np.full((self.n_envs, self.n_agents, 1), False, dtype=bool)
        info = []

        for i in range(self.n_envs):
            actions_i = actions[i]
            if self.gym_env:
                actions_i = actions_i.squeeze(0)     
            obs_i, reward_i, done_i, info_i = self.envs[i].step(actions_i)
            if self.gym_env:
                obs_i = np.expand_dims(obs_i, 0)
                done_i = np.expand_dims(done_i, 0)
            else:
                done_i = np.expand_dims(np.array(done_i), -1)
            reward_i = reward_i * self.n_agents    
            obs[i] = obs_i
            reward[i] = reward_i
            done[i] = done_i
            info.append(info_i)

        obs = torch.Tensor(obs)
        reward = torch.Tensor(reward)
        return obs, reward, done, info

    def reset(self):
        
        obs = np.zeros((self.n_envs, self.n_agents, self.obs_shape))
        
        for i in range(self.n_envs):
            obs_i = self.envs[i].reset()
            if self.gym_env:
                obs_i = np.expand_dims(obs_i, 0)
            obs[i] = obs_i
            
        obs = torch.Tensor(obs)
        return obs

    # Render the first env
    def render(self):
        self.envs[0].render()

    def seed(self, seed_value):
        for env in self.envs:
            env.seed(seed_value)   

class MultiAgentEnvWrapper:
    
    def __init__(self, env_id, seed):
        self.env_id = env_id
        self.seed = seed

        scenario = scenarios.load(env_id + ".py").Scenario()
        world = scenario.make_world()
        self.env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data, 
                seed_callback=scenario.seed, cam_range=scenario.world_radius)
        self.env.seed(seed)
        self.action_shape = 5
        self.obs_shape = self.env.observation_space[0].shape[0]
        self.n_agents = self.env.n   
        self.max_reward = 0

    def create_sim_env(self):
        scenario = scenarios.load(self.env_id + ".py").Scenario()
        world = scenario.make_world()
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data, 
                seed_callback=scenario.seed, cam_range=scenario.world_radius)
        env.seed(self.seed)

        return env

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)
