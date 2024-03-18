import numpy as np
import tensorflow as tf
from src.networks.cartpole.cartpole_policy_network import CartpolePolicyNetwork
from src.networks.cartpole.cartpole_value_network import CartpoleValueNetwork
from src.actor_critic.mcc import MccActorCritic

if __name__ == '__main__':
    np.random.seed(23)
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()

    # Load the weights of the Value network
    value = CartpoleValueNetwork(0.00055, restore_weights=True)
    # Load the weights of the Policy network
    policy = CartpolePolicyNetwork(2, 0.00001, restore_weights=True)
    # Initialize actor critic agent on the MountainCar environment
    agent = MccActorCritic(0.99, 0.00001, 0.00055, render=True, policy_nn=policy, value_nn=value)
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        agent.train()
