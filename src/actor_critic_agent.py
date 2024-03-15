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
from actor_critic_network import Network


Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

class ActorCriticAgent:
    def __init__(self, config):
        self.device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
        self.policy_network = Network(config['state_size'], config['action_size'], config['hidden_sizes'], is_policy=True).to(self.device)
        self.value_network = Network(config['state_size'], 1, config['hidden_sizes'], is_policy=False).to(self.device)
        self.optimizer_actor = optim.Adam(self.policy_network.parameters(), lr=config['lr_actor'])
        self.optimizer_critic = optim.Adam(self.value_network.parameters(), lr=config['lr_critic'])
        self.verbosity = config['verbosity']
        self.env_name = config['env_name']
        self.writer = SummaryWriter(f"runs/{config['experiment']}")
        self.gamma = config['gamma']


    def select_action(self, state):
        # without gradients --  test
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            probs = self.policy_network(state)
            m = Categorical(probs)
            action = m.sample()

        return action.item(), m.log_prob(action)


    def update_policy(self, transitions):
        loss_policy = 0
        loss_value = 0

        for transition in transitions:
            state, action, reward, next_state, done = transition
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            action = torch.tensor([action]).to(self.device)
            reward = torch.tensor([reward], dtype=torch.float).to(self.device)
            done = torch.tensor([done], dtype=torch.float).to(self.device)

            # Compute value loss
            predicted_value = self.value_network(state)
            next_predicted_value = self.value_network(next_state)
            expected_value = reward + self.gamma * next_predicted_value * (1 - done)
            loss_value += nn.MSELoss()(predicted_value, expected_value.detach())

            # compute policy loss
            probs = self.policy_network(state)
            m = Categorical(probs)
            log_prob = m.log_prob(action)
            advantage = expected_value - predicted_value.detach()
            loss_policy += (-log_prob * advantage).mean()

        # Backpropagate losses
        self.optimizer_actor.zero_grad()
        loss_policy.backward()
        self.optimizer_actor.step()

        self.optimizer_critic.zero_grad()
        loss_value.backward()
        self.optimizer_critic.step()

        return loss_policy.item(), loss_value.item()

    def save_models(self, path='models'):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.policy_network.state_dict(), os.path.join(path, f'{self.env_name}_policy_network.pth'))
        torch.save(self.value_network.state_dict(), os.path.join(path, f'{self.env_name}_value_network.pth'))

    def load_models(self, path='models', env_name='None'):
        if env_name == 'None':
            env_name = self.env_name
        self.policy_network.load_state_dict(torch.load(os.path.join(path, f'{env_name}_policy_network.pth'), map_location=self.device))
        self.value_network.load_state_dict(torch.load(os.path.join(path, f'{env_name}_value_network.pth'), map_location=self.device))

    def reinitialize_output_layers(self, new_action_size):
        """
        Reinitializes the output layers of both the policy and value networks
        for the new action size, freezing the hidden layers.
        """
        self.policy_network.reinitialize_output_layer(output_size=new_action_size)
        self.value_network.reinitialize_output_layer(output_size=1) # output_size is always 1 for the value network
        self.policy_network.to(self.device)
        self.value_network.to(self.device)


    def train(self, env_wrapper, max_episodes=1000, max_steps=500, reward_threshold=475.0, update_frequency=500):
        self.results = {'Episode': [], 'Reward': [], "Average_100": [], 'Solved': -1, 'Duration': 0, 'Loss': [], 'LossV': []}
        results = self.results
        start_time = time()
        episode_rewards = []
        total_steps = 0

        for episode in range(max_episodes):
            state = env_wrapper.reset()
            episode_reward = 0
            transitions = []

            for step in range(max_steps):
                action, log_prob = self.select_action(state)
                next_state, reward, done, _ = env_wrapper.step(action)
                transitions.append(Transition(state, action, reward, next_state, done))

                episode_reward += reward
                state = next_state

                # update policy
                if (len(transitions) >= update_frequency) or done:
                    loss_policy, loss_value = self.update_policy(transitions)
                    results['Loss'].append(loss_policy)
                    results['LossV'].append(loss_value)
                    transitions = [] # should we reset the transitions list here?

                if done:
                    break

            episode_rewards.append(episode_reward)

            results['Episode'].append(episode)
            results['Reward'].append(episode_reward)

            if len(episode_rewards) >= 100:
                avg_reward = sum(episode_rewards[-100:]) / 100
                results['Average_100'].append(avg_reward)
                if avg_reward > reward_threshold and results['Solved'] == -1:
                    results['Solved'] = episode
                    print(f"Solved at episode {episode} with average reward {avg_reward}.")
                    break
            else:
                results['Average_100'].append(sum(episode_rewards) / len(episode_rewards))

            if episode % self.verbosity == 0:
                print(f"Episode {episode}, Avg Reward: {results['Average_100'][-1]}, PLoss: {loss_policy}, VLoss: {loss_value}")

            # Log to TensorBoard
            self.writer.add_scalar("Reward", episode_reward, episode)
            self.writer.add_scalar("Average_100", results['Average_100'][-1], episode)
            self.writer.add_scalar("Loss_Policy", loss_policy, episode)
            self.writer.add_scalar("Loss_Value", loss_value, episode)

        results['Duration'] = time() - start_time
        self.writer.close()

        return results