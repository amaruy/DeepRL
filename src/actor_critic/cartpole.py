import gymnasium as gym
import numpy as np
import tensorflow as tf
import collections
from datetime import datetime
import time
# from torch.utils.tensorboard import SummaryWriter
# from logger import Logger
import json
import os
from src.networks.cartpole.cartpole_policy_network import CartpolePolicyNetwork
from src.networks.cartpole.cartpole_value_network import CartpoleValueNetwork


class CartpoleActorCritic:
    
    def __init__(self, discount_factor, policy_learning_rate, value_learning_rate, render=False, policy_nn=None, value_nn=None):
        # self.tb_writer = SummaryWriter("logs/cartpole/log-at2c-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
        # self.logger = Logger("logs/cartpole/log-at2c-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".csv")
        self.env = gym.make('CartPole-v1')

        self.state_size = 6
        self.action_size = 3
        self.env_state_size = 4
        self.env_action_size = 2

        self.max_episodes = 1500
        self.max_steps = 501

        self.render = render

        self.discount_factor = discount_factor
        self.policy_learning_rate = policy_learning_rate
        self.value_learning_rate = value_learning_rate

        # Initialize networks
        self.policy = CartpolePolicyNetwork(self.env_action_size, self.policy_learning_rate) if policy_nn is None else policy_nn
        self.value_network = CartpoleValueNetwork(self.value_learning_rate) if value_nn is None else value_nn
        
        
    def pad_with_zeros(self, v, pad_size):
        v_t = np.hstack((v, np.zeros(pad_size)))
        return v_t.reshape((1, v_t.shape[0]))


    def train(self, sess, save_model=False):
        tic = time.perf_counter()
        global_step = 0

        solved = False
        Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
        episode_rewards = np.zeros(self.max_episodes)
        average_rewards = 0.0

        for episode in range(self.max_episodes):
            state, _ = self.env.reset()
            state = self.pad_with_zeros(state, self.state_size - self.env_state_size)
            episode_transitions = []

            for step in range(self.max_steps):
                global_step += 1
                actions_distribution = sess.run(self.policy.actions_distribution, {self.policy.state: state})
                action = np.random.choice(np.arange(self.env_action_size), p=actions_distribution)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                next_state = self.pad_with_zeros(next_state, self.state_size - self.env_state_size)

                if self.render:
                    self.env.render()

                action_one_hot = np.zeros(self.action_size)
                action_one_hot[action] = 1
                episode_transitions.append(Transition(state=state, action=action_one_hot, reward=reward, next_state=next_state, done=done))
                episode_rewards[episode] += reward

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

                if done:
                    if episode > 98:
                        # Check if solved
                        average_rewards = np.mean(episode_rewards[(episode - 99):episode + 1])
                    print("Episode {} Reward: {} Average over 100 episodes: {}".format(episode, episode_rewards[episode], round(average_rewards, 2)))
                    # self.logger.write([episode, episode_rewards[episode], average_rewards, time.perf_counter() - tic])
                    # self.tb_writer.add_scalar('Rewards per episode', episode_rewards[episode], global_step=episode)
                    # self.tb_writer.add_scalar('Mean episode score over 100 consecutive episodes', average_rewards, global_step=episode)

                    if average_rewards > 475:
                        print(' Solved at episode: ' + str(episode))
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
    start = time.time()
    agent = CartpoleActorCritic(0.99, 0.0001, 0.0005, render=True)
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        agent.train(sess, save_model=True)
    end = time.time()
    print("Time taken: ", end - start)