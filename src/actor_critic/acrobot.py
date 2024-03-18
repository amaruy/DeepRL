import gymnasium as gym
import numpy as np
import tensorflow as tf
# from logger import Logger
import collections
from datetime import datetime
import time
# from torch.utils.tensorboard import SummaryWriter
import json
import os
from src.networks.acrobot.acrobot_policy_network import AcrobotPolicyNetwork
from src.networks.acrobot.acrobot_value_network import AcrobotValueNetwork


class AcrobotActorCritic:
    
    def __init__(self, discount_factor, policy_learning_rate, value_learning_rate, render=False, policy_nn=None, value_nn=None):
        np.random.seed(1)
        tf.compat.v1.disable_eager_execution()

        # self.tb_writer = SummaryWriter("logs/acrobot/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
        # self.logger = Logger("logs/acrobot/log-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".csv")

        self.env = gym.make('Acrobot-v1')

        self.state_size = 6
        self.action_size = 3

        self.max_episodes = 750
        self.max_steps = 501

        self.render = render

        self.discount_factor = discount_factor
        self.policy_learning_rate = policy_learning_rate
        self.value_learning_rate = value_learning_rate

        # Initialize the policy network
        self.policy = AcrobotPolicyNetwork(self.action_size, self.policy_learning_rate) if policy_nn is None else policy_nn
        # Initialize the value network
        self.value_network = AcrobotValueNetwork(self.value_learning_rate) if value_nn is None else value_nn

    def train(self, sess, save_model=False):
        tic = time.perf_counter()
        global_step = 0

        solved = False
        Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
        episode_rewards = []
        average_rewards = 0.0

        for episode in range(self.max_episodes):
            state = self.env.reset()[0]
            state = state.reshape((1, state.shape[0]))
            episode_transitions = []
            cumulative_reward = 0

            for step in range(self.max_steps):
                global_step += 1
                actions_distribution = sess.run(self.policy.actions_distribution, {self.policy.state: state})
                action = np.random.choice(np.arange(self.action_size), p=actions_distribution)
                next_state, reward, done, _, _ = self.env.step(action)
                next_state = next_state.reshape((1, self.state_size))

                if self.render:
                    self.env.render()

                action_one_hot = np.zeros(self.action_size)
                action_one_hot[action] = 1
                episode_transitions.append(Transition(state=state, action=action_one_hot, reward=reward, next_state=next_state, done=done))
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

                    print("Episode {} steps: {} Reward: {} Average over 100 episodes: {}".format(episode, step, np.round(episode_rewards[episode], 2), np.round(average_rewards, 2)))
                    # self.logger.write([episode, episode_rewards[episode], average_rewards, time.perf_counter() - tic])

                    # self.tb_writer.add_scalar('Rewards per episode', episode_rewards[episode], global_step=episode)
                    # self.tb_writer.add_scalar('Mean episode score over 100 consecutive episodes', average_rewards, global_step=episode)

                    # Check if solved
                    if average_rewards > -85 and average_rewards != 0:
                        print('Solved at episode: ' + str(episode))
                        solved = True
                    if save_model and solved:
                        self.policy.save_weights(sess)
                        self.value_network.save_weights(sess)
                    break
                state = next_state

            if solved:
                break

        # self.tb_writer.add_hparams({'discount_factor': self.discount_factor,
        #                        'policy_learning_rate': self.policy_learning_rate,
        #                        'value_learning_rate': self.value_learning_rate},
        #                       {'episodes_for_solution ': episode,
        #                        'average_rewards': average_rewards})


if __name__ == '__main__':
    np.random.seed(23)
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()

    agent = AcrobotActorCritic(0.99, 0.001, 0.001, render=True)
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        agent.train(sess, save_model=True)
