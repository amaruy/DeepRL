import gymnasium as gym
import numpy as np
import tensorflow as tf
import time
from src.networks.cartpole.prog_cartpole_policy_network import ProgCartpolePolicyNetwork
from src.networks.cartpole.prog_cartpole_value_network import ProgCartpoleValueNetwork
from src import config
import pickle
class CartpoleProgActorCritic:

    def __init__(self, discount_factor, policy_learning_rate, value_learning_rate, render=False, policy_nn=None, value_nn=None):
        self.env = gym.make(config.cartpole_env_name)
        self.state_size = config.state_size
        self.action_size = config.action_size
        self.env_state_size = config.cartpole_env_state_size
        self.env_action_size = config.cartpole_env_action_size
        self.max_episodes = config.cartpole_max_episodes
        self.max_steps = config.cartpole_max_steps
        self.render = render
        self.discount_factor = discount_factor
        self.policy_learning_rate = policy_learning_rate
        self.value_learning_rate = value_learning_rate
        self.policy = policy_nn or ProgCartpolePolicyNetwork(self.env_action_size, self.policy_learning_rate)
        self.value_network = value_nn or ProgCartpoleValueNetwork(self.value_learning_rate)
        self.metrics = {
            'policy_losses': [],
            'value_losses': [],
            'episode_rewards': [],
            'average_rewards': [],
            'hyperparameters': {}
        }
        self.save_metrics_path = config.prog_cartpole_run_results_path

    def pad_with_zeros(self, v, pad_size):
        v_t = np.hstack((np.squeeze(v), np.zeros(pad_size)))
        return v_t.reshape((1, v_t.shape[0]))
    
    def perform_action(self, sess, state):
        actions_distribution = sess.run(self.policy.actions_distribution, {self.policy.state: state})
        action = np.random.choice(np.arange(self.env_action_size), p=actions_distribution)
        next_state, reward, done, _, _ = self.env.step(action)
        next_state = self.pad_with_zeros(next_state, self.state_size - self.env_state_size)

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

        feed_dict = {self.value_network.state: state, self.value_network.R_t: td_target}
        _, v_loss = sess.run([self.value_network.optimizer, self.value_network.loss], feed_dict)

        feed_dict = {self.policy.state: state, self.policy.R_t: td_error, self.policy.action: action}
        _, p_loss = sess.run([self.policy.optimizer, self.policy.loss], feed_dict)

        return p_loss, v_loss

    def log_loss(self, p_loss, v_loss):
        self.metrics['policy_losses'].append(p_loss)
        self.metrics['value_losses'].append(v_loss)
            
    def log_rewards(self, cumulative_reward, average_rewards):
        # Log rewards per episode and mean episode score over 100 consecutive episodes
        self.metrics['episode_rewards'].append(cumulative_reward)
        self.metrics['average_rewards'].append(average_rewards)
            
    def log_train_score(self, score):
        # Log hyperparameters and metrics
        self.metrics['train_score'] = score
        
    def save_metrics(self):
        """Save the collected metrics to a json file."""
        with open(self.save_metrics_path, 'wb') as f:
            pickle.dump(self.metrics, f)
            
    def train(self, sess):
        global_step = 0
        solved = False
        episode_rewards = np.zeros(self.max_episodes)
        average_rewards = 0.0

        for episode in range(self.max_episodes):
            state, _ = self.env.reset()
            state = self.pad_with_zeros(state, self.state_size - self.env_state_size)

            for _ in range(self.max_steps):
                global_step += 1
                action, next_state, reward, done = self.perform_action(sess, state)
                episode_rewards[episode] += reward
                p_loss, v_loss = self.update_models(sess, state, action, reward, next_state, done)
                self.log_loss(p_loss, v_loss)
                if done:
                    if episode > 98:
                        # Check if solved
                        average_rewards = np.mean(episode_rewards[(episode - 99):episode + 1])
                    print("Episode {} Reward: {} Average over 100 episodes: {}".format(episode, episode_rewards[episode], round(average_rewards, 2)))
                    self.log_rewards(episode_rewards[episode], average_rewards)
                    if average_rewards > config.cartpole_avg_reward_thresh:
                        print(' Solved at episode: ' + str(episode))
                        solved = True
                    break
                state = next_state

            if solved:
                break

        train_score = {'discount_factor': self.discount_factor,
                                    'policy_learning_rate': self.policy_learning_rate,
                                    'value_learning_rate': self.value_learning_rate, 
                                    'episodes_for_solution ': episode,
                                    'average_rewards': average_rewards}
        
        self.log_train_score(train_score)
        self.save_metrics()


if __name__ == '__main__':
    np.random.seed(23)
    tf.compat.v1.disable_eager_execution()

    tf.compat.v1.reset_default_graph()
    agent = CartpoleProgActorCritic(0.99, 0.0003, 0.00072, render=True)
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        agent.train(sess)
