import tensorflow as tf
import json
from src import config

class MccPolicyNetwork:
    def __init__(self, env_action_size, learning_rate, name='policy_network'):
        self.state_size = config.state_size
        self.action_size = config.action_size
        self.env_action_size = env_action_size
        self.learning_rate = learning_rate
        self.size_first_hidden_layer = config.mcc_policy_hidden_1_size
        self.size_second_hidden_layer = config.mcc_policy_hidden_2_size
        
        with tf.compat.v1.variable_scope(name):
            self.init_weights()
            self.define_network_structure()
            self.define_loss_and_optimizer()

    
    def define_network_structure(self):
        """
        Defines the structure of the neural network.
        """
        self.state = tf.compat.v1.placeholder(tf.float32, [None, self.state_size], name="state")
        self.action = tf.compat.v1.placeholder(tf.int32, [self.action_size], name="action")
        self.R_t = tf.compat.v1.placeholder(tf.float32, name="total_rewards")

        self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
        self.A1 = tf.nn.elu(self.Z1)
        self.Z2 = tf.add(tf.matmul(self.A1, self.W2), self.b2)
        self.A2 = tf.nn.elu(self.Z2)
        self.output = tf.add(tf.matmul(self.A2, self.W3), self.b3)
        self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output[:, : self.env_action_size]))
        self.neg_log_prob = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(logits=self.output,
                                                                                    labels=self.action)
    def define_loss_and_optimizer(self):
        """
        Defines the loss function and the optimizer for training.
        """
        self.loss = tf.reduce_mean(self.neg_log_prob * self.R_t)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


    def init_weights(self):
        """
        Initializes weights and biases for the network.
        """
        self.W1 = tf.compat.v1.get_variable("W1", [self.state_size, self.size_first_hidden_layer], initializer=tf.initializers.GlorotUniform(seed=0))
        self.b1 = tf.compat.v1.get_variable("b1", [self.size_first_hidden_layer], initializer=tf.zeros_initializer())
        self.W2 = tf.compat.v1.get_variable("W2", [self.size_first_hidden_layer, self.size_second_hidden_layer], initializer=tf.initializers.GlorotUniform(seed=0))
        self.b2 = tf.compat.v1.get_variable("b2", [self.size_second_hidden_layer], initializer=tf.zeros_initializer())
        self.W3 = tf.compat.v1.get_variable("W3", [self.size_second_hidden_layer, self.action_size], initializer=tf.initializers.GlorotUniform(seed=0))
        self.b3 = tf.compat.v1.get_variable("b3", [self.action_size], initializer=tf.zeros_initializer())

    def save_weights(self, sess):
        """
        Saves the current weights of the network to a file.
        """
        W1, b1 = sess.run([self.W1, self.b1])
        W2, b2 = sess.run([self.W2, self.b2])
        W3, b3 = sess.run([self.W3, self.b3])
        weights = {'W1': W1.tolist(), 'b1': b1.tolist(), 'W2': W2.tolist(), 'b2': b2.tolist(), 'W3': W3.tolist(), 'b3': b3.tolist() }
        
        # Write the weights data to a JSON file
        with open(config.mcc_policy_weights, 'w') as f:
            json.dump(weights, f)