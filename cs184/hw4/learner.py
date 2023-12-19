from common import *
import torch
from torch import optim
import numpy as np
from dataset import ExpertData
from torch import distributions as pyd
import torch.nn as nn
import os
from stable_baselines3 import DQN

'''Imitation learning agents file.'''

class BC:
    def __init__(self, state_dim, action_dim, args):
        # Policy network setup
        self.policy = DiscretePolicy(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=args.lr)
        self.loss = nn.CrossEntropyLoss()

    def get_logits(self, states):
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states)
        return self.policy(states)

    def learn(self, expert_states, expert_actions):
        # Get logits under our policy
        logits = self.get_logits(expert_states)
        # Compute loss with target policy
        loss = self.loss(logits, expert_actions)
        # Backward step
        self.optimizer.zero_grad()
        loss.backward()
        # Update the policy
        self.optimizer.step()
        # Return current loss for saving
        return loss.item()

    def load(self, path):
        self.policy.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

class DAGGER:
    '''TBH can make a subclass of BC learner'''
    def __init__(self, state_dim, action_dim, args):
        self.policy = DiscretePolicy(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=args.lr)
        policy_path = os.path.join(args.expert_save_path, args.env.lower() + '_policy.pt')
        self.expert_policy = DQN.load(policy_path)
        self.loss = nn.CrossEntropyLoss()

    def rollout(self, env, num_steps):
        # Rollout for 'num_steps' steps in the environment. Reset if necessary.
        obs = torch.from_numpy(env.reset()).float()
        actions = torch.tensor([], dtype=int)
        for step in range(num_steps):
            # Save the observed state
            if step == 0:
                states = obs.unsqueeze(0)
            else:
                states = torch.cat([states, obs.unsqueeze(0)], dim=0)
            # Find expert action and save it
            expert_action = self.expert_policy.predict(obs)[0]
            expert_action = torch.from_numpy(np.array([expert_action]))
            actions = torch.cat([actions, expert_action])
            # Take action according to our policy
            logits = self.get_logits(obs)
            our_action = self.sample_from_logits(logits)
            obs, reward, done, info = env.step(our_action)
            obs = torch.from_numpy(obs).float()
            # Break if environment terminated
            if done:
                break
        return ExpertData(states, actions)
    
    def get_logits(self, states):
        return self.policy(states)

    def sample_from_logits(self, logits):
        probs = torch.softmax(logits, dim=0)
        action = torch.multinomial(probs, 1)
        return action.item()

    def learn(self, expert_states, expert_actions):
        # Get actions under current policy
        logits = self.get_logits(expert_states)
        # Compute loss with target policy
        loss = self.loss(logits, expert_actions)
        # Backward step
        self.optimizer.zero_grad()
        loss.backward()
        # Update the policy
        self.optimizer.step()
        # Return current loss for saving
        return loss.item()
    
    def load(self, path):
        self.policy.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.policy.state_dict(), path)
    