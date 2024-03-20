import tensorflow as tf
import json
from src import config

class ProgCartpoleValueNetwork:
    """
    Implements a progressive value network for the MountainCar environment using TensorFlow 1.x.
    """

    def __init__(self, learning_rate, name='value_network'):
        """
        Initializes the progressive value network.

        :param learning_rate: Learning rate for the optimizer.
        :param name: Name of the TensorFlow variable scope.
        """
        self.state_size = config.state_size
        self.learning_rate = learning_rate
        self.hidden_size = config.cartpole_prog_value_hidden_size

        with tf.compat.v1.variable_scope(name):
            self.state = tf.compat.v1.placeholder(tf.float32, [None, self.state_size], name="state")
            self.R_t = tf.compat.v1.placeholder(tf.float32, name="total_rewards")
            self.define_network_structure()
            self.define_loss_and_optimizer()

            
    def define_loss_and_optimizer(self):
        """
        Defines the loss and optimizer for the policy network.
        """
        self.loss = tf.compat.v1.losses.mean_squared_error(self.R_t, self.output)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def define_acrobot_value_network(self):
        """
        Defines the Acrobot value network.
        """
        self.restore_acrobat_value_weights()
        self.h1_k1 = tf.nn.elu(tf.add(tf.matmul(self.state, self.W1_k1), self.b1_k1))
        
    
    def define_mcc_value_network(self):
        """
        Defines the MountainCarContinuous value network.
        """        
        self.restore_mcc_value_weights()
        self.h1_k2 = tf.nn.elu(tf.add(tf.matmul(self.state, self.W1_k2), self.b1_k2))
        self.uk2_h1_k1 = tf.compat.v1.get_variable("uk2_h1_k1", [self.b1_k1.shape[0], self.b2_k2.shape[0]], initializer=tf.initializers.GlorotUniform(seed=0))
        self.h2_k2 = tf.nn.relu(tf.add(tf.add(tf.matmul(self.h1_k2, self.W2_k2), self.b2_k2), tf.matmul(self.h1_k1, self.uk2_h1_k1)))

    def define_cartpole_value_network(self):
        """
        Defines the Cartpole value network.
        """
        self.init_weights()
        self.h1 = tf.nn.elu(tf.add(tf.matmul(self.state, self.W1), self.b1))
        self.uk3_h1_k1 = tf.compat.v1.get_variable("uk3_h1_k1", [self.b1_k1.shape[0], self.b2.shape[0]], initializer=tf.initializers.GlorotUniform(seed=0))
        self.uk3_h1_k2 = tf.compat.v1.get_variable("uk3_h1_k2", [self.b1_k2.shape[0], self.b2.shape[0]], initializer=tf.initializers.GlorotUniform(seed=0))
        self.h2 = tf.nn.relu(tf.add(tf.add(tf.matmul(self.h1, self.W2), self.b2), tf.add(tf.matmul(self.h1_k1, self.uk3_h1_k1), tf.matmul(self.h1_k2, self.uk3_h1_k2))))
        self.uk3_h2_k2 = tf.compat.v1.get_variable("uk3_h2_k2", [self.b2_k2.shape[0], self.b3.shape[0]], initializer=tf.initializers.GlorotUniform(seed=0))
        self.output = tf.add(tf.add(tf.matmul(self.h2, self.W3), self.b3), tf.matmul(self.h2_k2, self.uk3_h2_k2))

    def define_network_structure(self):
        """
        Defines the neural network structure for the policy model.
        """
        self.define_acrobot_value_network()
        self.define_mcc_value_network()
        self.define_cartpole_value_network()


    def init_weights(self):
        """
        Initializes weights and biases for the network.
        """
        self.W1 = tf.compat.v1.get_variable("W1", [self.state_size, self.hidden_size], initializer=tf.initializers.GlorotUniform(seed=0))
        self.b1 = tf.compat.v1.get_variable("b1", [self.hidden_size], initializer=tf.zeros_initializer())
        self.W2 = tf.compat.v1.get_variable("W2", [self.hidden_size, self.hidden_size], initializer=tf.initializers.GlorotUniform(seed=0))
        self.b2 = tf.compat.v1.get_variable("b2", [self.hidden_size], initializer=tf.zeros_initializer())
        self.W3 = tf.compat.v1.get_variable("W3", [self.hidden_size, 1], initializer=tf.initializers.GlorotUniform(seed=0))
        self.b3 = tf.compat.v1.get_variable("b3", [1], initializer=tf.zeros_initializer())

    def restore_mcc_value_weights(self):
        """
        Restores the weights for the MountainCarContinuous value network.
        """
        with open(config.mcc_value_weights, 'r') as f:
            weights = json.load(f)
        self.W1_k2 = tf.compat.v1.get_variable("W1_k2", initializer=tf.constant(weights["W1"]), trainable=False)
        self.b1_k2 = tf.compat.v1.get_variable("b1_k2", initializer=tf.constant(weights["b1"]), trainable=False)
        self.W2_k2 = tf.compat.v1.get_variable("W2_k2", initializer=tf.constant(weights["W2"]), trainable=False)
        self.b2_k2 = tf.compat.v1.get_variable("b2_k2", initializer=tf.constant(weights["b2"]), trainable=False)
        self.W3_k2 = tf.compat.v1.get_variable("W3_k2", initializer=tf.constant(weights["W3"]), trainable=False)
        self.b3_k2 = tf.compat.v1.get_variable("b3_k2", initializer=tf.constant(weights["b3"]), trainable=False)

    def restore_acrobat_value_weights(self):
        """
        Restores the weights for the Acrobot value network.
        """
        with open(config.acrobot_value_weights, 'r') as f:
            weights = json.load(f)
        self.W1_k1 = tf.compat.v1.get_variable("W1_k1", initializer=tf.constant(weights["W1"]), trainable=False)
        self.b1_k1 = tf.compat.v1.get_variable("b1_k1", initializer=tf.constant(weights["b1"]), trainable=False)
        self.W2_k1 = tf.compat.v1.get_variable("W2_k1", initializer=tf.constant(weights["W2"]), trainable=False)
        self.b2_k1 = tf.compat.v1.get_variable("b2_k1", initializer=tf.constant(weights["b2"]), trainable=False)
        self.W3_k1 = tf.compat.v1.get_variable("W3_k1", initializer=tf.constant(weights["W2"]), trainable=False)
        self.b3_k1 = tf.compat.v1.get_variable("b3_k1", initializer=tf.constant(weights["b2"]), trainable=False)
