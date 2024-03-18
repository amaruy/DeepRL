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
    

class ProgMccPolicyNetwork:
    def __init__(self, env_action_size, learning_rate, name='policy_network'):
        self.state_size = 6
        self.action_size = 3
        self.learning_rate = learning_rate
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        self.W3 = None
        self.b3 = None

        with tf.compat.v1.variable_scope(name):
            self.state = tf.compat.v1.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.compat.v1.placeholder(tf.int32, [self.action_size], name="action")
            self.R_t = tf.compat.v1.placeholder(tf.float32, name="total_rewards")

            # First column - Cartpole NN
            self.restore_cartpole_policy_weights()
            self.h1_k1 = tf.nn.elu(tf.add(tf.matmul(self.state, self.W1_k1), self.b1_k1))
            self.h2_k1 = tf.nn.elu(tf.add(tf.matmul(self.h1_k1, self.W2_k1), self.b2_k1))

            # Second column - Acrobot NN
            self.restore_acrobat_policy_weights()
            self.h1_k2 = tf.nn.elu(tf.add(tf.matmul(self.state, self.W1_k2), self.b1_k2))

            # Third column - MountainCar NN
            self.init_weights()
            self.h1 = tf.nn.elu(tf.add(tf.matmul(self.state, self.W1), self.b1))
            self.uk1_k3_j1 = tf.compat.v1.get_variable("uk1_k3_j1", [self.b1_k1.shape[0], self.b2.shape[0]], initializer=tf.initializers.GlorotUniform(seed=0))
            self.h2 = tf.nn.relu(tf.add(tf.add(tf.matmul(self.h1, self.W2), self.b2), tf.matmul(self.h1_k1, self.uk1_k3_j1)))
            self.uk1_k3_j2 = tf.compat.v1.get_variable("uk1_k3_j2", [self.b2_k1.shape[0], self.b3.shape[0]], initializer=tf.initializers.GlorotUniform(seed=0))
            self.uk2_k3_j2 = tf.compat.v1.get_variable("uk2_k3_j2", [self.b1_k2.shape[0], self.b3.shape[0]], initializer=tf.initializers.GlorotUniform(seed=0))
            self.output = tf.add(tf.add(tf.matmul(self.h2, self.W3), self.b3), tf.add(tf.matmul(self.h2_k1, self.uk1_k3_j2), tf.matmul(self.h1_k2, self.uk2_k3_j2)))

            # Softmax probability distribution over actions
            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output[:, : env_action_size]))
            # Loss with negative log probability+
            self.neg_log_prob = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.action)
            self.loss = tf.reduce_mean(self.neg_log_prob * self.R_t)
            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def init_weights(self):
        hidden_1 = 32
        hidden_2 = 32
        self.W1 = tf.compat.v1.get_variable("W1", [self.state_size, hidden_1], initializer=tf.initializers.GlorotUniform(seed=0))
        self.b1 = tf.compat.v1.get_variable("b1", [hidden_1], initializer=tf.zeros_initializer())
        self.W2 = tf.compat.v1.get_variable("W2", [hidden_1, hidden_2], initializer=tf.initializers.GlorotUniform(seed=0))
        self.b2 = tf.compat.v1.get_variable("b2", [hidden_2], initializer=tf.zeros_initializer())
        self.W3 = tf.compat.v1.get_variable("W3", [hidden_2, self.action_size], initializer=tf.initializers.GlorotUniform(seed=0))
        self.b3 = tf.compat.v1.get_variable("b3", [self.action_size], initializer=tf.zeros_initializer())

    def restore_acrobat_policy_weights(self):
        with open(const.acrobot_policy_weights, 'r') as f:
            weights = json.load(f)
        self.W1_k2 = tf.compat.v1.get_variable("W1_k2", initializer=tf.constant(weights["W1"]), trainable=False)
        self.b1_k2 = tf.compat.v1.get_variable("b1_k2", initializer=tf.constant(weights["b1"]), trainable=False)
        self.W2_k2 = tf.compat.v1.get_variable("W2_k2", initializer=tf.constant(weights["W2"]), trainable=False)
        self.b2_k2 = tf.compat.v1.get_variable("b2_k2", initializer=tf.constant(weights["b2"]), trainable=False)

    def restore_cartpole_policy_weights(self):
        with open(const.cartpole_policy_weights, 'r') as f:
            weights = json.load(f)
        self.W1_k1 = tf.compat.v1.get_variable("W1_k1", initializer=tf.constant(weights["W1"]), trainable=False)
        self.b1_k1 = tf.compat.v1.get_variable("b1_k1", initializer=tf.constant(weights["b1"]), trainable=False)
        self.W2_k1 = tf.compat.v1.get_variable("W2_k1", initializer=tf.constant(weights["W2"]), trainable=False)
        self.b2_k1 = tf.compat.v1.get_variable("b2_k1", initializer=tf.constant(weights["b2"]), trainable=False)
        self.W3_k1 = tf.compat.v1.get_variable("W3_k1", initializer=tf.constant(weights["W3"]), trainable=False)
        self.b3_k1 = tf.compat.v1.get_variable("b3_k1", initializer=tf.constant(weights["b3"]), trainable=False)
