import gymnasium as gym
import numpy as np
import tensorflow as tf
import time
from src.networks.mcc.prog_mcc_policy_network import ProgMccPolicyNetwork
from src.networks.mcc.prog_mcc_value_network import ProgMccValueNetwork
from src import config
import pickle
import time
class MccProgActorCritic:
    """
    Class to train a MountainCarContinuous-v0 agent using Proximal Policy Optimization (PPO) algorithm.
    """
    def __init__(self, discount_factor, policy_learning_rate, value_learning_rate, render=False, policy_nn=None, value_nn=None,
                 save_metrics_path=None):

        self.env = gym.make(config.mcc_env_name)
        self.state_size = config.state_size
        self.action_size = config.action_size
        self.env_state_size = config.mcc_env_state_size
        self.env_action_size = config.mcc_env_action_size
        self.actions = config.mcc_actions
        self.max_episodes = config.mcc_max_episodes
        self.max_steps = config.mcc_max_steps
        self.render = render
        self.discount_factor = discount_factor
        self.policy_learning_rate = policy_learning_rate
        self.value_learning_rate = value_learning_rate
        self.policy = policy_nn or ProgMccPolicyNetwork(self.env_action_size, self.policy_learning_rate)
        self.value_network = value_nn or ProgMccValueNetwork(self.value_learning_rate)
        self.metrics = {
            'policy_losses': [],
            'value_losses': [],
            'episode_rewards': [],
            'average_rewards': [],
            'hyperparameters': {}
        }
        self.save_metrics_path = save_metrics_path

    def pad_with_zeros(self, v, pad_size):
        v_t = np.hstack((np.squeeze(v), np.zeros(pad_size)))
        return v_t.reshape((1, v_t.shape[0]))

    def scale_state(self, state):
        return [state[0] + 0.3, state[1] * 10]
    
    def perform_action(self, sess, state):
        actions_distribution = sess.run(self.policy.actions_distribution, {self.policy.state: state})
        action_index = np.random.choice(np.arange(self.env_action_size), p=actions_distribution)
        action = self.actions[action_index]
        next_state, reward, done, _, _ = self.env.step([action])
        next_state = self.pad_with_zeros(self.scale_state(next_state), self.state_size - self.env_state_size)
        if self.render:
            self.env.render()

        action_one_hot = np.zeros(self.action_size)
        action_one_hot[action_index] = 1
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
            
    def train(self):
        with tf.compat.v1.Session() as sess:
            start = time.time()
            sess.run(tf.compat.v1.global_variables_initializer())
            solved = False
            episode_rewards = []
            success_history = []
            average_rewards = 0.0
            global_step = 0

            for episode in range(self.max_episodes):
                state = self.scale_state(self.env.reset()[0])
                state = self.pad_with_zeros(state, self.state_size - self.env_state_size)
                cumulative_reward = 0
                
                for step in range(self.max_steps):
                    global_step += 1
                    action, next_state, reward, done = self.perform_action(sess, state)
                    cumulative_reward += reward
                    p_loss, v_loss = self.update_models(sess, state, action, reward, next_state, done)
                    self.log_loss(p_loss, v_loss)

                    if done or step == self.max_steps - 1:
                        episode_rewards.append(cumulative_reward)
                        average_rewards = np.mean(episode_rewards[-100:])
                        success_history.append(1 if cumulative_reward > 0 else 0)

                        print("Episode {} steps: {} Reward: {} Average over 100 episodes: {}, Average success: {}".format(episode, step, np.round(episode_rewards[episode], 2), np.round(average_rewards, 2), np.round(np.sum(success_history[-100:])/len(success_history[-100:]), 2)))
                        self.log_rewards(cumulative_reward, average_rewards)
                        if average_rewards > 85 and episode > 100:
                            print(' Solved at episode: ' + str(episode))
                            solved = True
                        if sum(success_history[-20:]) == 0 and episode > 10:
                            print(' Unlucky train: ' + str(episode))
                            solved = True
                        break

                    state = next_state

                if solved:
                    break

            train_score = {'discount_factor': self.discount_factor,
                                'policy_learning_rate': self.policy_learning_rate,
                                'value_learning_rate': self.value_learning_rate,
                                'episodes_for_solution ': episode,
                                'average_rewards': average_rewards,
                                'train_time': time.time() - start,
                                }
            self.log_train_score(train_score)
            self.save_metrics()


if __name__ == '__main__':
    np.random.seed(23)
    tf.compat.v1.disable_eager_execution()
    start = time.time()
    tf.compat.v1.reset_default_graph()
    agent = MccProgActorCritic(0.99, 0.00001, 0.00055, render=True, save_metrics_path=config.prog_cartpole_mcc_run_results_path)
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        agent.train()
    end = time.time()
    print("Time taken for transfer learning acrobot, cartpole car to mountain car: ", end - start)