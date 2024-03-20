import tensorflow as tf
import json
from src import config

class CartpoleValueNetwork:
    """
    Implements a value network for the Cartpole environment using TensorFlow 1.x.
    """
    
    def __init__(self, learning_rate, restore_weights=False, name='value_network'):
        """
        Initializes the value network with the option to restore weights from a file.

        :param learning_rate: Learning rate for the optimizer.
        :param restore_weights: Boolean indicating whether to restore weights.
        :param name: Name of the TensorFlow variable scope.
        """
        self.state_size = config.state_size
        self.learning_rate = learning_rate
        self.size_first_hidden_layer = config.cartpole_value_hidden_1_size
        self.size_second_hidden_layer = config.cartpole_value_hidden_2_size

        with tf.compat.v1.variable_scope(name):
            if restore_weights:
                self.restore_weights()
            else:
                self.init_weights()

            self.define_network_structure()
            self.define_loss_and_optimizer()

    def define_network_structure(self):
        """
        Defines the structure of the neural network.
        """
        self.state = tf.compat.v1.placeholder(tf.float32, [None, self.state_size], name="state")
        self.R_t = tf.compat.v1.placeholder(tf.float32, name="total_rewards")
        self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
        self.A1 = tf.nn.elu(self.Z1)
        self.Z2 = tf.add(tf.matmul(self.A1, self.W2), self.b2)
        self.A2 = tf.nn.elu(self.Z2)
        self.output = tf.add(tf.matmul(self.A2, self.W3), self.b3)

    def define_loss_and_optimizer(self):
        """
        Defines the loss function and the optimizer for training.
        """
        self.loss = tf.compat.v1.losses.mean_squared_error(self.R_t, self.output)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def init_weights(self):
        """
        Initializes weights and biases for the network.
        """
        self.W1 = tf.compat.v1.get_variable("W1", [self.state_size, self.size_first_hidden_layer], initializer=tf.initializers.GlorotUniform(seed=0))
        self.b1 = tf.compat.v1.get_variable("b1", [self.size_first_hidden_layer], initializer=tf.zeros_initializer())
        self.W2 = tf.compat.v1.get_variable("W2", [self.size_first_hidden_layer, self.size_second_hidden_layer], initializer=tf.initializers.GlorotUniform(seed=0))
        self.b2 = tf.compat.v1.get_variable("b2", [self.size_second_hidden_layer], initializer=tf.zeros_initializer())
        self.W3 = tf.compat.v1.get_variable("W3", [self.size_second_hidden_layer, 1], initializer=tf.initializers.GlorotUniform(seed=0))
        self.b3 = tf.compat.v1.get_variable("b3", [1], initializer=tf.zeros_initializer())

    def save_weights(self, sess):
        """
        Saves the current weights of the network to a file.
        """
        weights = {
            'W1': sess.run(self.W1).tolist(), 'b1': sess.run(self.b1).tolist(),
            'W2': sess.run(self.W2).tolist(), 'b2': sess.run(self.b2).tolist(),
            'W3': sess.run(self.W3).tolist(), 'b3': sess.run(self.b3).tolist()
        }
        
        with open(config.cartpole_value_weights, 'w') as f:
            json.dump(weights, f)

    def restore_weights(self):
        """
        Restores the weights of the network from a file.
        """
        with open(config.cartpole_value_weights, 'r') as f:
            weights = json.load(f)
            
        self.W1 = tf.compat.v1.get_variable("W1", initializer=tf.constant(weights["W1"]))
        self.b1 = tf.compat.v1.get_variable("b1", initializer=tf.constant(weights["b1"]))
        self.W2 = tf.compat.v1.get_variable("W2", initializer=tf.constant(weights["W2"]))
        self.b2 = tf.compat.v1.get_variable("b2", initializer=tf.constant(weights["b2"]))
        self.W3 = tf.compat.v1.get_variable("W3", [64, 1], initializer=tf.initializers.GlorotUniform(seed=0))
        self.b3 = tf.compat.v1.get_variable("b3", [1], initializer=tf.zeros_initializer())
