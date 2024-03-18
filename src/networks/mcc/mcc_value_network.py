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


class MccValueNetwork:

    def __init__(self, learning_rate, name='value_network'):
        self.state_size = 6
        self.learning_rate = learning_rate

        with tf.compat.v1.variable_scope(name):
            self.state = tf.compat.v1.placeholder(tf.float32, [None, self.state_size], name="state")
            self.R_t = tf.compat.v1.placeholder(tf.float32, name="total_rewards")

            self.init_weights()
            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.elu(self.Z1)
            self.Z2 = tf.add(tf.matmul(self.A1, self.W2), self.b2)
            self.A2 = tf.nn.elu(self.Z2)
            self.output = tf.add(tf.matmul(self.A2, self.W3), self.b3)

            self.loss = tf.compat.v1.losses.mean_squared_error(self.R_t, self.output)
            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def init_weights(self):
        hidden_1 = 64
        hidden_2 = 64
        self.W1 = tf.compat.v1.get_variable("W1", [self.state_size, hidden_1], initializer=tf.initializers.GlorotUniform(seed=0))
        self.b1 = tf.compat.v1.get_variable("b1", [hidden_1], initializer=tf.zeros_initializer())
        self.W2 = tf.compat.v1.get_variable("W2", [hidden_1, hidden_2], initializer=tf.initializers.GlorotUniform(seed=0))
        self.b2 = tf.compat.v1.get_variable("b2", [hidden_2], initializer=tf.zeros_initializer())
        self.W3 = tf.compat.v1.get_variable("W3", [hidden_2, 1], initializer=tf.initializers.GlorotUniform(seed=0))
        self.b3 = tf.compat.v1.get_variable("b3", [1], initializer=tf.zeros_initializer())

    def save_weights(self, sess):
        W1, b1 = sess.run([self.W1, self.b1])
        W2, b2 = sess.run([self.W2, self.b2])
        W3, b3 = sess.run([self.W3, self.b3])
        weights = {'W1': W1.tolist(), 'b1': b1.tolist(), 'W2': W2.tolist(), 'b2': b2.tolist(), 'W3': W3.tolist(),
                    'b3': b3.tolist()}
        
        # Check if the directory exists, create it if it does not
        if not os.path.exists(const.WEIGHTS_PATH):
            os.makedirs(const.WEIGHTS_PATH)

        # Write the weights data to a JSON file
        with open(const.mcc_value_weights, 'w') as f:
            json.dump(weights, f)
            