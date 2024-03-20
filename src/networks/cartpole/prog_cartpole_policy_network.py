import tensorflow as tf
import json
import src.config as config


class ProgCartpolePolicyNetwork:
    """
    Implements a progressive policy network for the Cartpole environment using TensorFlow 1.x.
    """
    def __init__(self, env_action_size, learning_rate, name='policy_network'):
        """
        Initializes the progressive policy network.

        :param learning_rate: Learning rate for the optimizer.
        :param name: Name of the TensorFlow variable scope.
        """
        self.state_size = config.state_size
        self.action_size = config.action_size
        self.learning_rate = learning_rate
        self.env_action_size = env_action_size
        self.hidden_size = config.cartpole_prog_policy_hidden_size


        with tf.compat.v1.variable_scope(name):
            self.define_network_structure()
            self.define_loss_and_optimizer()
            
    def define_acrobot_policy_network(self):
        """
        Defines the policy network for the Acrobot environment using TensorFlow 1.x.
        """
        self.restore_acrobat_policy_weights()
        self.h1_k1 = tf.nn.elu(tf.add(tf.matmul(self.state, self.W1_k1), self.b1_k1))
        
    
    def define_mcc_policy_network(self):
        """
        Defines the policy network for the MountainCarContinuous environment using TensorFlow 1.x.
        """
        self.restore_mcc_policy_weights()
        self.h1_k2 = tf.nn.elu(tf.add(tf.matmul(self.state, self.W1_k2), self.b1_k2))
        self.uk2_h1_k1 = tf.compat.v1.get_variable("uk2_h1_k1", [self.b1_k1.shape[0], self.b2_k2.shape[0]], initializer=tf.initializers.GlorotUniform(seed=0))
        self.h2_k2 = tf.nn.relu(tf.add(tf.add(tf.matmul(self.h1_k2, self.W2_k2), self.b2_k2), tf.matmul(self.h1_k1, self.uk2_h1_k1)))


    def define_cartpole_policy_network(self):
        """
        Defines the policy network for the Cartpole environment using TensorFlow 1.x.
        """
        self.init_weights()
        self.h1 = tf.nn.elu(tf.add(tf.matmul(self.state, self.W1), self.b1))
        self.uk3_h1_k1 = tf.compat.v1.get_variable("uk3_h1_k1", [self.b1_k1.shape[0], self.b2.shape[0]], initializer=tf.initializers.GlorotUniform(seed=0))
        self.uk3_h1_k2 = tf.compat.v1.get_variable("uk3_h1_k2", [self.b1_k2.shape[0], self.b2.shape[0]], initializer=tf.initializers.GlorotUniform(seed=0))
        self.h2 = tf.nn.relu(tf.add(tf.add(tf.matmul(self.h1, self.W2), self.b2), tf.add(tf.matmul(self.h1_k1, self.uk3_h1_k1), tf.matmul(self.h1_k2, self.uk3_h1_k2))))
        self.uk3_h2_k2 = tf.compat.v1.get_variable("uk3_h2_k2", [self.b2_k2.shape[0], self.b3.shape[0]], initializer=tf.initializers.GlorotUniform(seed=0))
        self.output = tf.add(tf.add(tf.matmul(self.h2, self.W3), self.b3), tf.matmul(self.h2_k2, self.uk3_h2_k2))

        # Network Output
        self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output[:, : self.env_action_size]))
        self.neg_log_prob = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.action)  
                  
    def define_network_structure(self):
        """
        Defines the neural network structure for the policy model.
        """
        self.state = tf.compat.v1.placeholder(tf.float32, [None, self.state_size], name="state")
        self.action = tf.compat.v1.placeholder(tf.int32, [self.action_size], name="action")
        self.R_t = tf.compat.v1.placeholder(tf.float32, name="total_rewards")

        self.define_acrobot_policy_network()
        self.define_mcc_policy_network()
        self.define_cartpole_policy_network()

            
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
        self.W1 = tf.compat.v1.get_variable("W1", [self.state_size, self.hidden_size], initializer=tf.initializers.GlorotUniform(seed=0))
        self.b1 = tf.compat.v1.get_variable("b1", [self.hidden_size], initializer=tf.zeros_initializer())
        self.W2 = tf.compat.v1.get_variable("W2", [self.hidden_size, self.hidden_size], initializer=tf.initializers.GlorotUniform(seed=0))
        self.b2 = tf.compat.v1.get_variable("b2", [self.hidden_size], initializer=tf.zeros_initializer())
        self.W3 = tf.compat.v1.get_variable("W3", [self.hidden_size, self.action_size], initializer=tf.initializers.GlorotUniform(seed=0))
        self.b3 = tf.compat.v1.get_variable("b3", [self.action_size], initializer=tf.zeros_initializer())

    def restore_acrobat_policy_weights(self):
        """
        Restores the weights of the Acrobot policy network.
        """
        with open(config.acrobot_policy_weights, 'r') as f:
            weights = json.load(f)
        self.W1_k1 = tf.compat.v1.get_variable("W1_k1", initializer=tf.constant(weights["W1"]), trainable=False)
        self.b1_k1 = tf.compat.v1.get_variable("b1_k1", initializer=tf.constant(weights["b1"]), trainable=False)
        self.W2_k1 = tf.compat.v1.get_variable("W2_k1", initializer=tf.constant(weights["W2"]), trainable=False)
        self.b2_k1 = tf.compat.v1.get_variable("b2_k1", initializer=tf.constant(weights["b2"]), trainable=False)

    def restore_mcc_policy_weights(self):
        """
        Restores the weights of the MountainCarContinuous policy network.
        """
        with open(config.mcc_policy_weights, 'r') as f:
            weights = json.load(f)
        self.W1_k2 = tf.compat.v1.get_variable("W1_k2", initializer=tf.constant(weights["W1"]), trainable=False)
        self.b1_k2 = tf.compat.v1.get_variable("b1_k2", initializer=tf.constant(weights["b1"]), trainable=False)
        self.W2_k2 = tf.compat.v1.get_variable("W2_k2", initializer=tf.constant(weights["W2"]), trainable=False)
        self.b2_k2 = tf.compat.v1.get_variable("b2_k2", initializer=tf.constant(weights["b2"]), trainable=False)
        self.W3_k2 = tf.compat.v1.get_variable("W3_k2", initializer=tf.constant(weights["W3"]), trainable=False)
        self.b3_k2 = tf.compat.v1.get_variable("b3_k2", initializer=tf.constant(weights["b3"]), trainable=False)
        