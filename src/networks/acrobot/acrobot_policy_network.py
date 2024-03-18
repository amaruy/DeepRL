import gymnasium as gym
import numpy as np
import tensorflow as tf
from datetime import datetime
import time
# from torch.utils.tensorboard import SummaryWriter
# from logger import Logger
import json
import os
import src.config as const


class AcrobotPolicyNetwork:
    def __init__(self, env_action_size, learning_rate, restore_weights=False, name='policy_network'):
        self.state_size = 6
        self.action_size = 3
        self.learning_rate = learning_rate

        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

        with tf.compat.v1.variable_scope(name):
            self.state = tf.compat.v1.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.compat.v1.placeholder(tf.int32, [self.action_size], name="action")
            self.R_t = tf.compat.v1.placeholder(tf.float32, name="total_rewards")

            if restore_weights:
                self.restore_weights()
            else:
                self.init_weights()

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)

            # Softmax probability distribution over actions
            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output[:, : env_action_size]))

            # Loss with negative log probability
            self.neg_log_prob = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(logits=self.output,
                                                                                        labels=self.action)
            self.loss = tf.reduce_mean(self.neg_log_prob * self.R_t)
            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def init_weights(self):
        self.W1 = tf.compat.v1.get_variable("W1", [self.state_size, 12], initializer=tf.initializers.GlorotUniform(seed=0))
        self.b1 = tf.compat.v1.get_variable("b1", [12], initializer=tf.zeros_initializer())
        self.W2 = tf.compat.v1.get_variable("W2", [12, self.action_size], initializer=tf.initializers.GlorotUniform(seed=0))
        self.b2 = tf.compat.v1.get_variable("b2", [self.action_size], initializer=tf.zeros_initializer())

    def save_weights(self, sess):
        W1, b1 = sess.run([self.W1, self.b1])
        W2, b2 = sess.run([self.W2, self.b2])
        weights = {'W1': W1.tolist(), 'b1': b1.tolist(), 'W2': W2.tolist(), 'b2': b2.tolist()}
        
        # Check if the directory exists, create it if it does not
        if not os.path.exists(const.WEIGHTS_PATH):
            os.makedirs(const.WEIGHTS_PATH)

        # Write the weights data to a JSON file
        with open(const.acrobot_policy_weights, 'w') as f:
            json.dump(weights, f)


    def restore_weights(self):
        with open(const.acrobot_policy_weights, 'r') as f:
            weights = json.load(f)
        self.W1 = tf.compat.v1.get_variable("W1", initializer=tf.constant(weights["W1"]))
        self.b1 = tf.compat.v1.get_variable("b1", initializer=tf.constant(weights["b1"]))
        self.W2 = tf.compat.v1.get_variable("W2", [12, self.action_size], initializer=tf.initializers.GlorotUniform(seed=0))
        self.b2 = tf.compat.v1.get_variable("b2", [self.action_size], initializer=tf.zeros_initializer())