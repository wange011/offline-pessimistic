import torch
import torch.nn.functional as F
import torch.multiprocessing as multiprocessing

import numpy as np

import time
import os
import json

from agents import safari
from envs import env

class MultiTrainer:

    def __init__(self, configs, device):
        self.configs = configs
        self.device = device
        
    def train(self, max_parallel):
        for i in range(0, len(self.configs), max_parallel):
            parallel_configs = self.configs[i : min(i + max_parallel, len(self.configs))]

            processes = []
            for config in parallel_configs:
                p = multiprocessing.Process(target=self.run_trainer, args=(config, self.device))
                processes.append(p)
                p.start()
            
            for p in processes:
                p.join()
    
    def run_trainer(self, config, device):
        trainer = Trainer(config, device)
        trainer.train()

class Trainer:

    def __init__(self, config, device):
        env_ = env.MultiEnv(config.main.scenario_name, config.main.n_envs, config.main.seed, gym_env=config.main.gym_env)

        if config.main.alg == "safari":
            self.agents = safari.SAFARI(env_.envs[0], config, device)
        else:
            raise NotImplementedError

        self.env = env_
        self.config = config
        self.device = device
    
        self.dir_main = os.path.join(os.getcwd(), "results", config.main.dir_main)

        if 'dir_data' in self.config.main:
            self.dir_data = os.path.join(os.getcwd(), "data", config.main.dir_data)

        os.makedirs(self.dir_main, exist_ok=True)

    def create_log(self):
        with open(os.path.join(self.dir_main, 'log.csv'), 'w') as f:
            f.write('update, episode, steps, time, reward\n')
    
    def write_to_log(self, update, episode, steps, time, reward):
        with open(os.path.join(self.dir_main, 'log.csv'), 'a') as f:
            f.write('{}, {}, {}, {}, {}\n'.format(update, episode, steps, time, reward))

    def train_offline(self, tuning=False):
        if not tuning:
            self.create_log()
        
        start_time = time.time()
        self.agents.load_data(self.dir_data)

        self.agents.update()

        rewards = [self.agents.run_episode() for i in range(self.config.main.n_eval_episodes)]
        reward = np.mean(rewards)

        if not tuning:
            self.write_to_log(self.config.alg.N, self.config.alg.N, 
                            self.config.alg.N * self.config.alg.H, time.time() - start_time, reward)
        
        self.agents.save(self.dir_main)
        return reward
