import numpy as np
import tensorflow as tf
from src.networks.acrobot.acrobot_policy_network import AcrobotPolicyNetwork
from src.networks.acrobot.acrobot_value_network import AcrobotValueNetwork
from src.actor_critic.cartpole import CartpoleActorCritic

if __name__ == '__main__':
    np.random.seed(23)
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()

    # Load the weights for the acrobots value and policy networks
    value = AcrobotValueNetwork(0.0008, restore_weights=True)
    policy = AcrobotPolicyNetwork(2, 0.0005, restore_weights=True)
    
    # Initialize actor critic agent on the Cartpole environment
    agent = CartpoleActorCritic(0.99, 0.0005, 0.0008, render=True, policy_nn=policy, value_nn=value)
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        agent.train(sess)
