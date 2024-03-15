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


class ContinuousActorCriticAgent(ActorCriticAgent):
    def __init__(self, config):
        super().__init__(config)
        # Ensure the policy network is suitable for continuous action spaces
        self.policy_network = Network(config['state_size'], config['action_size'], config['hidden_sizes'], discrete=False).to(self.device)
        
    def select_action(self, state):
        # Adjust for continuous action space
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_mean, action_std = self.policy_network(state)
        dist = Normal(action_mean, action_std.exp())
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action.cpu().detach().numpy(), log_prob

    def update_policy(self, transitions):
        loss_policy = 0
        loss_value = 0

        for transition in transitions:
            state, action, reward, next_state, done = transition
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            action = torch.FloatTensor(action).unsqueeze(0).to(self.device)
            reward = torch.FloatTensor([reward]).to(self.device)
            done = torch.FloatTensor([done]).to(self.device)

            # Compute value loss as before
            predicted_value = self.value_network(state)
            next_predicted_value = self.value_network(next_state)
            expected_value = reward + self.gamma * next_predicted_value * (1 - done)
            loss_value += nn.MSELoss()(predicted_value.squeeze(), expected_value.detach().squeeze())

            # Adjust policy loss computation for continuous action space
            action_mean, action_std = self.policy_network(state)
            dist = Normal(action_mean, action_std.exp())
            log_prob = dist.log_prob(action).sum(-1, keepdim=True)
            advantage = expected_value - predicted_value.detach()
            loss_policy += (-log_prob * advantage).mean()

        # Perform backpropagation and optimization as before
        self.optimizer_actor.zero_grad()
        loss_policy.backward()
        self.optimizer_actor.step()

        self.optimizer_critic.zero_grad()
        loss_value.backward()
        self.optimizer_critic.step()

        return loss_policy.item(), loss_value.item()
    