import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Categorical, Normal
import numpy as np
import os
from collections import namedtuple
from time import time
from torch.utils.tensorboard import SummaryWriter


######################################################################
#2. Reshape the environment wrapper to handle the action space
class EnvironmentWrapper:
    def __init__(self, env, target_state_size=6, target_action_size=3):
        self.env = env
        self.target_state_size = target_state_size
        self.target_action_size = target_action_size
        self.action_space = env.action_space.n if isinstance(env.action_space, gym.spaces.Discrete) else env.action_space.shape[0]

    def reset(self):
        state, _ = self.env.reset()
        # Pad state to match target_state_size
        padded_state = np.append(state, np.zeros(self.target_state_size - len(state)))
        return padded_state

    def step(self, action):
        action = min(self.action_space-1, max(0, action))
        
        state, reward, done, _, info = self.env.step(action)
        # Pad state to match target_state_size
        padded_state = np.append(state, np.zeros(self.target_state_size - len(state)))
        return padded_state, reward, done, info

    def action_padding(self, action):
        one_hot_action = np.zeros(self.target_action_size)
        one_hot_action[action] = 1
        return one_hot_action