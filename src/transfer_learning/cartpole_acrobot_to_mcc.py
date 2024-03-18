import gymnasium as gym
import numpy as np
import tensorflow as tf
from datetime import datetime
import time
# from torch.utils.tensorboard import SummaryWriter
# from logger import Logger
import json
import os
from src.networks.mcc.prog_mcc_policy_network import ProgMccPolicyNetwork
from src.networks.mcc.prog_mcc_value_network import ProgMccValueNetwork


class MccProgActorCritic:
    
    def pad_with_zeros(self, v, pad_size):
        v_t = np.hstack((np.squeeze(v), np.zeros(pad_size)))
        return v_t.reshape((1, v_t.shape[0]))

    def scale_state(self, state):
        return [state[0] + 0.3, state[1] * 10]

    def __init__(self, discount_factor, policy_learning_rate, value_learning_rate, render=False, policy_nn=None, value_nn=None):
        # self.tb_writer = SummaryWriter("logs/mcc/log-prog-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
        # self.logger = Logger("logs/mcc/log-prog-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".csv")
        self.env = gym.make('MountainCarContinuous-v0')

        self.state_size = 6
        self.action_size = 3
        self.env_state_size = 2
        self.env_action_size = 2
        self.actions = [-1, 1]

        self.max_episodes = 1500
        self.max_steps = 1000

        self.render = render

        self.discount_factor = discount_factor
        self.policy_learning_rate = policy_learning_rate
        self.value_learning_rate = value_learning_rate

        # Initialize the policy network
        self.policy = ProgMccPolicyNetwork(self.env_action_size, self.policy_learning_rate) if policy_nn is None else policy_nn
        # Initialize the value network
        self.value_network = ProgMccValueNetwork(self.value_learning_rate) if value_nn is None else value_nn

    def train(self):
        tic = time.perf_counter()
        global_step = 0
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            solved = False
            episode_rewards = []
            success_history = []
            average_rewards = 0.0

            for episode in range(self.max_episodes):
                state = self.scale_state(self.env.reset()[0])
                state = self.pad_with_zeros(state, self.state_size - self.env_state_size)

                cumulative_reward = 0
                for step in range(self.max_steps):
                    global_step += 1
                    actions_distribution = sess.run(self.policy.actions_distribution, {self.policy.state: state})
                    action_index = np.random.choice(np.arange(self.env_action_size), p=actions_distribution)
                    action = self.actions[action_index]
                    next_state, reward, done, _, _ = self.env.step([action])
                    next_state = self.pad_with_zeros(self.scale_state(next_state), self.state_size - self.env_state_size)
                    if self.render:
                        self.env.render()

                    action_one_hot = np.zeros(self.action_size)
                    action_one_hot[action_index] = 1
                    cumulative_reward += reward

                    current_value = sess.run(self.value_network.output, {self.value_network.state: state})
                    next_value = sess.run(self.value_network.output, {self.value_network.state: next_state})
                    td_target = reward + (1 - done) * self.discount_factor * next_value
                    td_error = td_target - current_value

                    feed_dict = {self.value_network.state: state, self.value_network.R_t: td_target}
                    _, v_loss = sess.run([self.value_network.optimizer, self.value_network.loss], feed_dict)

                    feed_dict = {self.policy.state: state, self.policy.R_t: td_error, self.policy.action: action_one_hot}
                    _, loss = sess.run([self.policy.optimizer, self.policy.loss], feed_dict)

                    # self.tb_writer.add_scalar('Policy network loss', loss, global_step=global_step)
                    # self.tb_writer.add_scalar('Value network loss', v_loss, global_step=global_step)

                    if done or step == self.max_steps - 1:
                        episode_rewards.append(cumulative_reward)
                        average_rewards = np.mean(episode_rewards[-100:])
                        success_history.append(1 if cumulative_reward > 0 else 0)

                        print("Episode {} steps: {} Reward: {} Average over 100 episodes: {}, Average success: {}".format(episode, step, np.round(episode_rewards[episode], 2), np.round(average_rewards, 2), np.round(np.sum(success_history[-100:])/len(success_history[-100:]), 2)))
                        # self.logger.write([episode, episode_rewards[episode], average_rewards, time.perf_counter() - tic])
                        # self.tb_writer.add_scalar('Rewards per episode', episode_rewards[episode], global_step=episode)
                        # self.tb_writer.add_scalar('Mean episode score over 100 consecutive episodes', average_rewards, global_step=episode)

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

            # self.tb_writer.add_hparams({'discount_factor': self.discount_factor,
            #                        'policy_learning_rate': self.policy_learning_rate,
            #                        'value_learning_rate': self.value_learning_rate},
            #                       {'episodes_for_solution ': episode,
            #                        'average_rewards': average_rewards,
            #                        'Average success': np.round(np.sum(success_history[-100:])/len(success_history[-100:]), 2)})


if __name__ == '__main__':
    np.random.seed(23)
    tf.compat.v1.disable_eager_execution()

    tf.compat.v1.reset_default_graph()
    agent = MccProgActorCritic(0.99, 0.00001, 0.00055, render=True)
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        agent.train()
        