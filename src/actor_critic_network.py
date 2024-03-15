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
# 1. Define the Policy (Actor) and Value (Critic) Networks
class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[64, 64], is_policy=True):
        super(Network, self).__init__()
        self.hidden_sizes = hidden_sizes
        layers = [nn.Linear(input_size, hidden_sizes[0]), nn.ReLU()]
        for i in range(len(hidden_sizes)-1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        self.network = nn.Sequential(*layers)
        self.is_policy = is_policy

    def forward(self, x):
        if self.is_policy:
            return nn.Softmax(dim=-1)(self.network(x))
        else:
            return self.network(x)

    def reinitialize_output_layer(self, output_size, freeze_hidden_layers=True):
        if freeze_hidden_layers:
            for param in self.network[:-1].parameters():
                param.requires_grad = False
        
        self.network[-1] = nn.Linear(self.hidden_sizes[-1], output_size)
        nn.init.normal_(self.network[-1].weight, mean=0., std=0.1)
        nn.init.constant_(self.network[-1].bias, 0)
        if self.is_policy:
            self.network.add_module("softmax", nn.Softmax(dim=-1))