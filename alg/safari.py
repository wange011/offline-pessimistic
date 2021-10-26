import torch
import torch.nn.functional as F

import numpy as np
import math

import copy
import os
import json

from collections import defaultdict, Counter, deque

class SAFARI:
    def __init__(self, env, config, device):
        self.N = config.alg.N
        self.H = config.alg.H
        self.n_agents = env.n_agents

        self.kernel = {}
        self.k_matrix = {}
        self.phi = [{} for i in range(self.H)]
        self.q_hat = [{} for i in range(self.H)]
        self.pi_hat = [{} for i in range(self.H)]
        self.v = [{} for i in range(self.H)]

        self.omega = []
        self.actions = []
        self.rewards = []
        self.seen_omega = set()

        self.env = env
        self.max_reward = 25

        self.possible_actions = [i for i in range(env.action_shape)]

        self.config = config
        self.params = config.alg

    def totuple(self, x):
        try:
            return tuple(self.totuple(i) for i in x)
        except TypeError:
            return x

    def process_obs(self, obs):
        """
        obs: [n_agents, obs_shape]
        output: ([obs_shape], [obs_shape])
        """

        # Let the first agent be the representative
        obs_rep = np.array(obs)[0]
        obs_rep = np.around(obs_rep, decimals=1)

        obs_rest = np.array(obs)[1:]
        d = np.around(d, decimals=1)

        return (self.totuple(obs_rep), self.totuple(d))
    
    def load_data(self, dir_data):
        self.dir_data = dir_data

        obs_file = open(os.path.join(self.dir_data, 'obs.txt'), "r")
        lines = obs_file.readlines()

        for i, line in enumerate(lines):
            if i >= self.N:
                break
            obs_episode = json.loads(line)
            omega_episode = []
            for obs in obs_episode:
                omega_h = self.process_obs(obs)
                omega_episode.append(omega_h)
                self.seen_omega.add(omega_h)
            self.omega.append(omega_episode)

        if len(self.omega) < self.N:
            print("N should be: {}".format(len(self.omega)))
            self.N = len(self.omega)

        actions_file = open(os.path.join(self.dir_data, 'actions.txt'), "r")
        lines = actions_file.readlines()

        for i, line in enumerate(lines):
            if i >= self.N:
                break
            actions_episode = json.loads(line)
            self.actions.append(actions_episode)
        
        rewards_file = open(os.path.join(self.dir_data, 'rewards.txt'), "r")
        lines = rewards_file.readlines()

        for i, line in enumerate(lines):
            if i >= self.N:
                break
            rewards_episode = json.loads(line)
            rewards_episode[0] += self.max_reward
            self.rewards.append(rewards_episode)

    def get_kernel(self, omega, action, omega_, action_):
        if (omega, self.totuple(action), omega_, self.totuple(action_)) in self.kernel:
            return self.kernel[(omega, self.totuple(action), omega_, self.totuple(action_))]

        x_1 = np.concatenate(([omega[0]], [omega[1]], [action]))
        x_2 = np.concatenate(([omega_[0]], [omega_[1]], [action_]))

        k = np.exp(- np.linalg.norm(x_1 - x_2, ord=2) ** 2 / self.params.sigma_kernel ** 2)

        self.kernel[(omega, self.totuple(action), omega_, self.totuple(action_))] = k
        return k

    def get_k_matrix(self, h):
        idx_h = h - 1

        if idx_h in self.k_matrix:
            return self.k_matrix[idx_h]
        
        k_matrix = []
        for i in range(self.N):
            row = []
            for j in range(self.N):
                kernel = self.get_kernel(self.omega[i][idx_h], self.actions[i][idx_h],
                                         self.omega[j][idx_h], self.actions[j][idx_h])
                row.append(kernel)
            k_matrix.append(row)
        k_matrix = np.array(k_matrix)

        self.k_matrix[idx_h] = k_matrix
        return k_matrix

    def get_lambda(self, h):
        k_matrix = self.get_k_matrix(h)
        lamb = k_matrix + self.params.lamb * np.identity(k_matrix.shape[0])
        return lamb

    def get_phi(self, h, omega, action):
        idx_h = h - 1

        if (omega, self.totuple(action)) in self.phi[idx_h]:
            return self.phi[idx_h][(omega, self.totuple(action))]

        phi = [self.get_kernel(self.omega[i][idx_h], self.actions[i][idx_h], omega, action) for i in range(self.N)]

        self.phi[idx_h][(omega, self.totuple(action))] = phi
        return phi

    def get_alpha(self, h):
        k_matrix = self.get_k_matrix(h)
        m = k_matrix + self.params.lamb * np.identity(k_matrix.shape[0])
        m_inv = np.linalg.inv(m)
        
        idx_h = h - 1

        returns = np.array([self.rewards[i][idx_h] for i in range(self.N)])
        if h < self.H:
            values = np.array([self.get_v(h + 1, self.omega[i][idx_h + 1]) for i in range(self.N)])
            returns += values
        
        alpha = np.matmul(m_inv, returns)
        return alpha

    def get_q_tilde(self, h, omega, action):
        phi = self.get_phi(h, omega, action)
        alpha = self.get_alpha(h)

        q = np.matmul(phi, alpha)
        return q

    def get_gamma(self, h, omega, action):
        kernel = self.get_kernel(omega, action, omega, action)
        phi = self.get_phi(h, omega, action)
        lamb = self.get_lambda(h)
        lamb_inv = np.linalg.inv(lamb)

        gamma = self.params.beta * self.params.lamb ** (-0.5)
        gamma *= (kernel - np.dot(np.matmul(phi, lamb_inv), phi)) ** (-0.5)
        return gamma

    def get_q_hat(self, h, omega, action):
        idx_h = h - 1

        if (omega, self.totuple(action)) in self.q_hat[idx_h]:
            return self.q_hat[idx_h][(omega, self.totuple(action))]

        q_tilde = self.get_q_tilde(h, omega, action)
        gamma = self.get_gamma(h, omega, action)

        max_reward = self.max_reward
        q = min(q_tilde - gamma, max_reward * self.H - h + max_reward)
        q = max(0, q)
        
        self.q_hat[idx_h][(omega, self.totuple(action))] = q
        return q

    def get_pi_hat(self, h, omega, eval=False):
        """
        output: 1
        """
        idx_h = h - 1
        if eval and omega not in self.pi_hat[idx_h]:
            print("New obs during run_episode")
            return np.random.randint(0, self.env.action_shape)

        if omega in self.pi_hat[idx_h]:
            return self.pi_hat[idx_h]

        max_q = None
        max_action = None
        for action in self.possible_actions:
            q = self.get_q_hat(h, omega, action)
            
            if max_q is None or q > max_q:
                max_q = q
                max_action = action

        self.pi_hat[idx_h][omega] = max_action 

        return max_action

    def get_v(self, h, omega):
        if h == self.H + 1:
            return 0
        
        idx_h = h - 1
        
        if omega in self.v[idx_h]:
            return self.v[idx_h][omega]
        
        action = self.get_pi_hat(h, omega)
        q = self.get_q_hat(h, omega, action)

        v = q
        self.v[idx_h][omega] = v
        return v
    
    def update(self):
        for h in reversed(range(1, self.H)):
            for omega in self.seen_omega:
                self.get_v(h, omega)
            print("Completed {}".format(h))
    
    def run_episode(self):
        obs = self.env.reset()
        omega = self.process_obs(obs)

        reward = 0
        for h in range(1, self.H + 1):
            action = self.get_pi_hat(h, omega, eval=True)
            actions = [action for i in range(self.env.n_agents)]

            next_obs, rewards, done, info = self.env.step(actions)
            reward += rewards[0]

            next_omega = self.process_obs(next_obs)
            omega = next_omega

        return reward
    
    def save(self, dir_main):
        v = []
        for dict_ in self.v:
            v_dict = {}
            for key, value in dict_.items():
                v_dict[str(key)] = value
            v.append(v_dict)

        pi_hat = []
        for dict_ in self.pi_hat:
            pi_hat_dict = {}
            for key, value in dict_.items():
                pi_hat_dict[str(key)] = value
            pi_hat.append(pi_hat_dict)

        data = {
            'v': v,
            'pi': pi_hat
        }

        with open(os.path.join(dir_main, 'safari.json'), 'w') as f:
            json.dump(data, f)