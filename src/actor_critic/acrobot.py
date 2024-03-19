import gymnasium as gym
import numpy as np
import tensorflow as tf
import collections
from datetime import datetime
import time
from tensorflow.summary import create_file_writer
from tensorboard.plugins.hparams import api as hp
import json
import os
from src.networks.acrobot.acrobot_policy_network import AcrobotPolicyNetwork
from src.networks.acrobot.acrobot_value_network import AcrobotValueNetwork
from src import config
import pickle

class AcrobotActorCritic:
    
    def __init__(self, discount_factor, policy_learning_rate, value_learning_rate, render=False, policy_nn=None, value_nn=None):
        np.random.seed(1)
        tf.compat.v1.disable_eager_execution()
         
        self.env = gym.make(config.acrobot_env_name)
        self.state_size = config.state_size
        self.action_size = config.action_size
        self.max_episodes = config.acrobot_max_episodes
        self.max_steps = config.acrobot_max_steps
        self.render = render
        self.discount_factor = discount_factor
        self.policy_learning_rate = policy_learning_rate
        self.value_learning_rate = value_learning_rate
        self.policy = policy_nn or AcrobotPolicyNetwork(self.action_size, self.policy_learning_rate)
        self.value_network = value_nn or AcrobotValueNetwork(self.value_learning_rate)
        # Initialize containers for metrics
        self.metrics = {
            'policy_losses': [],
            'value_losses': [],
            'episode_rewards': [],
            'average_rewards': [],
            'hyperparameters': {}
        }
        self.save_metrics_path = config.acrobot_run_results_path

    def perform_action(self, sess, state):
        actions_distribution = sess.run(self.policy.actions_distribution, {self.policy.state: state})
        action = np.random.choice(np.arange(self.action_size), p=actions_distribution)
        next_state, reward, done, _, _ = self.env.step(action)
        next_state = next_state.reshape((1, self.state_size))

        if self.render:
            self.env.render()

        action_one_hot = np.zeros(self.action_size)
        action_one_hot[action] = 1
        return action_one_hot, next_state, reward, done
    
    def update_models(self, sess, state, action, reward, next_state, done):
        current_value = sess.run(self.value_network.output, {self.value_network.state: state})
        next_value = sess.run(self.value_network.output, {self.value_network.state: next_state})
        td_target = reward + (1 - done) * self.discount_factor * next_value
        td_error = td_target - current_value
        # Update value network
        feed_dict = {self.value_network.state: state, self.value_network.R_t: td_target}
        _, v_loss = sess.run([self.value_network.optimizer, self.value_network.loss], feed_dict)
        # Update policy network
        feed_dict = {self.policy.state: state, self.policy.R_t: td_error, self.policy.action: action}
        _, loss = sess.run([self.policy.optimizer, self.policy.loss], feed_dict)
        return loss, v_loss
    
    def log_loss(self, p_loss, v_loss):
        self.metrics['policy_losses'].append(p_loss)
        self.metrics['value_losses'].append(v_loss)
            
    def log_rewards(self, cumulative_reward, average_rewards):
        # Log rewards per episode and mean episode score over 100 consecutive episodes
        self.metrics['episode_rewards'].append(cumulative_reward)
        self.metrics['average_rewards'].append(average_rewards)
            
    def log_hyperparameters_and_final_score(self, hparams):
        # Log hyperparameters and metrics
        self.metrics['hyperparameters'] = hparams
        
        
    def save_metrics(self):
        """Save the collected metrics to a json file."""
        with open(self.save_metrics_path, 'wb') as f:
            pickle.dump(self.metrics, f)


    def train(self, sess, save_model=False):
        episode_rewards = []
        average_rewards = 0.0
        global_step = 0
        solved = False

        for episode in range(self.max_episodes):
            state = self.env.reset()[0]
            state = state.reshape((1, state.shape[0]))
            cumulative_reward = 0

            for step in range(self.max_steps):
                action, next_state, reward, done = self.perform_action(sess, state)
                cumulative_reward += reward
                global_step += 1
                v_loss, p_loss = self.update_models(sess, state, action, reward, next_state, done)

                # Log policy network loss and value network loss
                self.log_loss(p_loss, v_loss)
                    
                if done or step == self.max_steps - 1:
                    episode_rewards.append(cumulative_reward)
                    average_rewards = np.mean(episode_rewards[-100:])

                    print(f"Episode {episode} Reward: {episode_rewards[episode]} Average over 100 episodes: {average_rewards}")
                    self.log_rewards(cumulative_reward, average_rewards)
                    # Check if solved
                    if average_rewards > config.acrobot_avg_reward_thresh and average_rewards != 0:
                        print('Solved at episode: ' + str(episode))
                        solved = True
                    if save_model and solved:
                        self.policy.save_weights(sess)
                        self.value_network.save_weights(sess)
                    break
                state = next_state

            if solved:
                break

        hparams_and_final_score = {
            'discount_factor': self.discount_factor,
            'policy_learning_rate': self.policy_learning_rate,
            'value_learning_rate': self.value_learning_rate,
            'episodes_for_solution': episode,
            'average_rewards': average_rewards,
        }

        self.log_hyperparameters_and_final_score(hparams_and_final_score)
        self.save_metrics()


if __name__ == '__main__':
    np.random.seed(23)
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()

    agent = AcrobotActorCritic(0.99, 0.001, 0.001, render=True)
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        agent.train(sess, save_model=True)
