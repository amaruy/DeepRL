import tensorflow as tf
import json
import src.config as config

class ProgMccPolicyNetwork:
    """
    Implements a progressive policy network for the MountainCar environment using TensorFlow 1.x.
    """
    def __init__(self, env_action_size, learning_rate, name='policy_network'):
        """
        Initializes the progressive policy network.

        :param learning_rate: Learning rate for the optimizer.
        :param name: Name of the TensorFlow variable scope.
        """
        self.state_size = config.state_size
        self.action_size = config.action_size
        self.env_action_size = env_action_size
        self.learning_rate = learning_rate
        self.size_first_hidden_layer = config.mcc_prog_policy_hidden_size
        self.size_second_hidden_layer = config.mcc_prog_policy_hidden_size

        with tf.compat.v1.variable_scope(name):
            self.define_network_structure()
            self.define_loss_and_optimizer()
            
            
    def define_cartpole_policy_network(self):
        """
        Defines the policy network for the Cartpole environment using TensorFlow 1.x.
        """
        self.restore_cartpole_policy_weights()
        self.h1_k1 = tf.nn.elu(tf.add(tf.matmul(self.state, self.W1_k1), self.b1_k1))
        self.h2_k1 = tf.nn.elu(tf.add(tf.matmul(self.h1_k1, self.W2_k1), self.b2_k1))

            
    def define_acrobot_policy_network(self):
        """
        Defines the policy network for the Acrobot environment using TensorFlow 1.x.
        """
        self.restore_acrobat_policy_weights()
        self.h1_k2 = tf.nn.elu(tf.add(tf.matmul(self.state, self.W1_k2), self.b1_k2))


    def define_mcc_policy_network(self):
        """
        Defines the policy network for the MountainCarContinuous environment using TensorFlow 1.x.
        """
        self.init_weights()
        self.h1 = tf.nn.elu(tf.add(tf.matmul(self.state, self.W1), self.b1))
        self.uk1_k3_j1 = tf.compat.v1.get_variable("uk1_k3_j1", [self.b1_k1.shape[0], self.b2.shape[0]], initializer=tf.initializers.GlorotUniform(seed=0))
        self.h2 = tf.nn.relu(tf.add(tf.add(tf.matmul(self.h1, self.W2), self.b2), tf.matmul(self.h1_k1, self.uk1_k3_j1)))
        self.uk1_k3_j2 = tf.compat.v1.get_variable("uk1_k3_j2", [self.b2_k1.shape[0], self.b3.shape[0]], initializer=tf.initializers.GlorotUniform(seed=0))
        self.uk2_k3_j2 = tf.compat.v1.get_variable("uk2_k3_j2", [self.b1_k2.shape[0], self.b3.shape[0]], initializer=tf.initializers.GlorotUniform(seed=0))
        self.output = tf.add(tf.add(tf.matmul(self.h2, self.W3), self.b3), tf.add(tf.matmul(self.h2_k1, self.uk1_k3_j2), tf.matmul(self.h1_k2, self.uk2_k3_j2)))
        self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output[:, : self.env_action_size]))
        self.neg_log_prob = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.action)
    
            
    def define_network_structure(self):
        """
        Defines the neural network structure for the policy model.
        """
        self.state = tf.compat.v1.placeholder(tf.float32, [None, self.state_size], name="state")
        self.action = tf.compat.v1.placeholder(tf.int32, [self.action_size], name="action")
        self.R_t = tf.compat.v1.placeholder(tf.float32, name="total_rewards")

        # First column - Cartpole NN
        self.define_cartpole_policy_network()
        # Second column - Acrobot NN
        self.define_acrobot_policy_network()
        # Third column - MountainCar NN
        self.define_mcc_policy_network()


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


    def restore_acrobat_policy_weights(self):
        """
        Restores the weights for the Acrobot policy network.
        """
        with open(config.acrobot_policy_weights, 'r') as f:
            weights = json.load(f)
        self.W1_k2 = tf.compat.v1.get_variable("W1_k2", initializer=tf.constant(weights["W1"]), trainable=False)
        self.b1_k2 = tf.compat.v1.get_variable("b1_k2", initializer=tf.constant(weights["b1"]), trainable=False)
        self.W2_k2 = tf.compat.v1.get_variable("W2_k2", initializer=tf.constant(weights["W2"]), trainable=False)
        self.b2_k2 = tf.compat.v1.get_variable("b2_k2", initializer=tf.constant(weights["b2"]), trainable=False)


    def restore_cartpole_policy_weights(self):
        """
        Restores the weights for the Cartpole policy network.
        """
        with open(config.cartpole_policy_weights, 'r') as f:
            weights = json.load(f)
        self.W1_k1 = tf.compat.v1.get_variable("W1_k1", initializer=tf.constant(weights["W1"]), trainable=False)
        self.b1_k1 = tf.compat.v1.get_variable("b1_k1", initializer=tf.constant(weights["b1"]), trainable=False)
        self.W2_k1 = tf.compat.v1.get_variable("W2_k1", initializer=tf.constant(weights["W2"]), trainable=False)
        self.b2_k1 = tf.compat.v1.get_variable("b2_k1", initializer=tf.constant(weights["b2"]), trainable=False)
        self.W3_k1 = tf.compat.v1.get_variable("W3_k1", initializer=tf.constant(weights["W3"]), trainable=False)
        self.b3_k1 = tf.compat.v1.get_variable("b3_k1", initializer=tf.constant(weights["b3"]), trainable=False)
