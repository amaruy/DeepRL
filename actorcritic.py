import gymnasium
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm
import numpy as np
import os
from collections import namedtuple
from time import time

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Environment Wrapper to Handle Different Environments
class EnvironmentWrapper:
    def __init__(self, env_name, state_pad, action_pad):
        self.env = gym.make(env_name)
        self.state_pad = state_pad
        self.action_pad = action_pad
        
    def reset(self):
        state, _ = self.env.reset()
        return np.append(state, np.zeros(self.state_pad - len(state)))
    
    def step(self, action):
        state, reward, done, _, info = self.env.step(action)
        state = np.append(state, np.zeros(self.state_pad - len(state)))
        return state, reward, done, info

    def render(self):
        self.env.render()

# Define the Policy (Actor) and Value (Critic) Networks
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 12),
            nn.ReLU(),
            nn.Linear(12, output_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.network(x)

class ValueNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super(ValueNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.network(x)

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

class ActorCriticAgent:
    def __init__(self, state_size, action_size, lr_actor=0.001, lr_critic=0.0005):
        self.policy_network = PolicyNetwork(state_size, action_size).to(device)
        self.value_network = ValueNetwork(state_size).to(device)
        self.optimizer_actor = optim.Adam(self.policy_network.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.value_network.parameters(), lr=lr_critic)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        probs = self.policy_network(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def update_policy(self, transitions, gamma=0.99):
        loss_policy = 0
        loss_value = 0

        for transition in transitions:
            state, action, reward, next_state, done = transition

            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            action = torch.tensor(action).view(1, -1).to(device)
            reward = torch.tensor(reward).float().to(device)
            done = torch.tensor(done).float().to(device)

            # Compute value loss
            predicted_value = self.value_network(state)
            next_predicted_value = self.value_network(next_state)
            expected_value = reward + gamma * next_predicted_value * (1 - done)
            loss_value += nn.MSELoss()(predicted_value, expected_value.detach())

            # Compute policy loss
            _, log_prob = self.select_action(state)
            advantage = expected_value - predicted_value.detach()
            loss_policy += -log_prob * advantage

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
        torch.save(self.policy_network.state_dict(), os.path.join(path, 'policy_network.pth'))
        torch.save(self.value_network.state_dict(), os.path.join(path, 'value_network.pth'))

    def load_models(self, path='models'):
        self.policy_network.load_state_dict(torch.load(os.path.join(path, 'policy_network.pth'), map_location=device))
        self.value_network.load_state_dict(torch.load(os.path.join(path, 'value_network.pth'), map_location=device))



    def train(self, env_wrapper, max_episodes=1000, max_steps=500, reward_threshold=475.0):
        results = {'Episode': [], 'Reward': [], "Average_100": [], 'Solved': -1, 'Duration': 0, 'Loss': [], 'LossV': []}
        start_time = time()
        episode_rewards = []

        for episode in tqdm(range(max_episodes)):
            state = env_wrapper.reset()
            episode_reward = 0
            transitions = []

            for step in range(max_steps):
                action, log_prob = self.select_action(state)
                next_state, reward, done, _ = env_wrapper.step(action)
                transitions.append(Transition(state, action, reward, next_state, done))

                episode_reward += reward
                state = next_state

                if done:
                    break

            loss_policy, loss_value = self.update_policy(transitions)
            episode_rewards.append(episode_reward)
            results['Episode'].append(episode)
            results['Reward'].append(episode_reward)
            results['Loss'].append(loss_policy)
            results['LossV'].append(loss_value)

            if len(episode_rewards) >= 100:
                avg_reward = sum(episode_rewards[-100:]) / 100
                results['Average_100'].append(avg_reward)
                if avg_reward > reward_threshold and results['Solved'] == -1:
                    results['Solved'] = episode
                    print(f"Solved at episode {episode} with average reward {avg_reward}.")
                    break
            else:
                results['Average_100'].append(sum(episode_rewards) / len(episode_rewards))

            if episode % 10 == 0:
                print(f"Episode {episode}, Reward: {episode_reward}, Avg Reward: {results['Average_100'][-1]}, Policy Loss: {loss_policy}, Value Loss: {loss_value}")

        results['Duration'] = time() - start_time

        return results

