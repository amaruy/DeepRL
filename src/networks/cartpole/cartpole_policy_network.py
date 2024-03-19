import tensorflow as tf
import json
from src import config

class CartpolePolicyNetwork:
    """
    Implements a policy network for the Cartpole environment using TensorFlow 1.x.
    """
    
    def __init__(self, env_action_size, learning_rate, restore_weights=False, name='policy_network'):
        """
        Initializes the policy network with optional weight restoration.

        :param env_action_size: Number of actions in the environment.
        :param learning_rate: Learning rate for the optimizer.
        :param restore_weights: Flag indicating whether to restore weights.
        :param name: Name of the TensorFlow variable scope.
        """
        self.state_size = config.state_size
        self.action_size = config.action_size
        self.learning_rate = learning_rate
        self.env_action_size = env_action_size

        # Use hidden layer sizes from the config
        self.size_first_hidden_layer = config.cartpole_policy_hidden_1_size
        self.size_second_hidden_layer = config.cartpole_policy_hidden_2_size

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
        self.A1 = tf.nn.elu(self.Z1)
        self.Z2 = tf.add(tf.matmul(self.A1, self.W2), self.b2)
        self.A2 = tf.nn.elu(self.Z2)
        self.output = tf.add(tf.matmul(self.A2, self.W3), self.b3)

        # define softmax for action distribution
        self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output[:, : self.env_action_size]))
        self.neg_log_prob = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(logits=self.output,
                                                                                        labels=self.action)
    def define_loss_and_optimizer(self):
        """
        Defines the loss and optimizer for the policy network.
        """
        self.loss = tf.reduce_mean(self.neg_log_prob * self.R_t)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def init_weights(self):
        """
        Initializes the weights of the policy network.
        """
        self.W1 = tf.compat.v1.get_variable("W1", [self.state_size, self.size_first_hidden_layer], initializer=tf.initializers.GlorotUniform(seed=0))
        self.b1 = tf.compat.v1.get_variable("b1", [self.size_first_hidden_layer], initializer=tf.zeros_initializer())
        self.W2 = tf.compat.v1.get_variable("W2", [self.size_first_hidden_layer, self.size_second_hidden_layer], initializer=tf.initializers.GlorotUniform(seed=0))
        self.b2 = tf.compat.v1.get_variable("b2", [self.size_second_hidden_layer], initializer=tf.zeros_initializer())
        self.W3 = tf.compat.v1.get_variable("W3", [self.size_second_hidden_layer, self.action_size], initializer=tf.initializers.GlorotUniform(seed=0))
        self.b3 = tf.compat.v1.get_variable("b3", [self.action_size], initializer=tf.zeros_initializer())

    def save_weights(self, sess):
        """
        Saves the network's weights to a file.
        """
        weights = {
            'W1': sess.run(self.W1).tolist(), 
            'b1': sess.run(self.b1).tolist(),
            'W2': sess.run(self.W2).tolist(), 
            'b2': sess.run(self.b2).tolist(),
            'W3': sess.run(self.W3).tolist(), 
            'b3': sess.run(self.b3).tolist()
        }
        
        with open(config.cartpole_policy_weights, 'w') as f:
            json.dump(weights, f)
    
    
    def restore_weights(self):
        """
        Restores the weights of the network from a file.
        """
        with open(config.cartpole_policy_weights, 'r') as f:
            weights = json.load(f)
        self.W1 = tf.compat.v1.get_variable("W1", initializer=tf.constant(weights["W1"]))
        self.b1 = tf.compat.v1.get_variable("b1", initializer=tf.constant(weights["b1"]))
        self.W2 = tf.compat.v1.get_variable("W2", initializer=tf.constant(weights["W2"]))
        self.b2 = tf.compat.v1.get_variable("b2", initializer=tf.constant(weights["b2"]))
        self.W3 = tf.compat.v1.get_variable("W3", [self.size_second_hidden_layer, self.action_size], initializer=tf.initializers.GlorotUniform(seed=0))
        self.b3 = tf.compat.v1.get_variable("b3", [self.action_size], initializer=tf.zeros_initializer())
