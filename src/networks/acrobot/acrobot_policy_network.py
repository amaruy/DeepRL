import tensorflow as tf
import json
import os
from src import config

class AcrobotPolicyNetwork:
    """
    Implements a policy network for the Acrobot environment using TensorFlow 1.x.
    """
    
    def __init__(self, env_action_size, learning_rate, restore_weights=False, name='policy_network'):
        """
        Initializes the policy network with optional weight restoration.

        :param learning_rate: Learning rate for the optimizer.
        :param restore_weights: Flag indicating whether to restore weights.
        :param name: Name of the TensorFlow variable scope.
        """
        self.state_size = config.state_size
        self.action_size = config.action_size
        self.env_action_size = env_action_size
        self.learning_rate = learning_rate
        self.hidden_layer_size = config.acrobot_policy_hidden_layer_size
        
        with tf.compat.v1.variable_scope(name):
            if restore_weights:
                self.restore_weights()
            else:
                self.init_weights()
                
            self.define_network_structure()
            self.define_loss_and_optimizer()

    def define_network_structure(self):
        """
        Defines the neural network structure for the policy model.
        """
        self.state = tf.compat.v1.placeholder(tf.float32, [None, self.state_size], name="state")
        self.action = tf.compat.v1.placeholder(tf.int32, [self.action_size], name="action")
        self.R_t = tf.compat.v1.placeholder(tf.float32, name="total_rewards")
        self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
        self.A1 = tf.nn.relu(self.Z1)
        self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)
        self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output[:, : self.env_action_size]))

    def define_loss_and_optimizer(self):
        """
        Defines the loss function and optimizer for the policy network.
        """
        self.neg_log_prob = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.action)
        self.loss = tf.reduce_mean(self.neg_log_prob * self.R_t)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def init_weights(self):
        """
        Initializes the weights and biases for the network.
        """
        self.W1 = tf.compat.v1.get_variable("W1", [self.state_size, self.hidden_layer_size], initializer=tf.initializers.GlorotUniform(seed=0))
        self.b1 = tf.compat.v1.get_variable("b1", [self.hidden_layer_size], initializer=tf.zeros_initializer())
        self.W2 = tf.compat.v1.get_variable("W2", [self.hidden_layer_size, self.action_size], initializer=tf.initializers.GlorotUniform(seed=0))
        self.b2 = tf.compat.v1.get_variable("b2", [self.action_size], initializer=tf.zeros_initializer())

    def save_weights(self, sess):
        """
        Saves the network's weights to a file.

        :param sess: The TensorFlow session.
        """
        weights = {
            'W1': sess.run(self.W1).tolist(), 
            'b1': sess.run(self.b1).tolist(),
            'W2': sess.run(self.W2).tolist(), 
            'b2': sess.run(self.b2).tolist()
        }
        
        if not os.path.exists(config.WEIGHTS_PATH):
            os.makedirs(config.WEIGHTS_PATH)

        with open(config.acrobot_policy_weights, 'w') as f:
            json.dump(weights, f)

    def restore_weights(self):
        """
        Restores the network's weights from a file.
        """
        try:
            with open(config.acrobot_policy_weights, 'r') as f:
                weights = json.load(f)
            # Rest of your code
        except FileNotFoundError:
            print(f"File {config.acrobot_policy_weights} not created yet.")

        self.W1 = tf.compat.v1.get_variable("W1", initializer=tf.constant(weights["W1"]))
        self.b1 = tf.compat.v1.get_variable("b1", initializer=tf.constant(weights["b1"]))
        self.W2 = tf.compat.v1.get_variable("W2", [self.hidden_layer_size, self.action_size], initializer=tf.initializers.GlorotUniform(seed=0))
        self.b2 = tf.compat.v1.get_variable("b2", [self.action_size], initializer=tf.zeros_initializer())
