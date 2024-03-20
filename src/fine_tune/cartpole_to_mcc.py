import numpy as np
import tensorflow as tf
from src.networks.cartpole.cartpole_policy_network import CartpolePolicyNetwork
from src.networks.cartpole.cartpole_value_network import CartpoleValueNetwork
from src.actor_critic.mcc import MccActorCritic
import time
import os
from src import config

def fine_tune_cartpole_to_mcc(value_lr=0.00055, policy_lr=0.00001, discount_factor=0.99, 
                                      restore_weights=True, render=True, save_metrics_path=None):
    """
    Fine-tune Cartpole networks to MountainCar environment using an actor-critic agent.

    Parameters:
    - value_lr: Learning rate for the value network.
    - policy_lr: Learning rate for the policy network.
    - discount_factor: Discount factor for future rewards.
    - restore_weights: Whether to restore weights for the networks.
    - render: Whether to render the environment.
    """
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()

    # Load the weights for the Cartpole value and policy networks
    value = CartpoleValueNetwork(value_lr, restore_weights=restore_weights)
    policy = CartpolePolicyNetwork(config.cartpole_env_action_size, policy_lr, restore_weights=restore_weights)
    
    # Initialize actor critic agent on the MountainCar environment
    agent = MccActorCritic(discount_factor, policy_lr, value_lr, render=render, 
                        policy_nn=policy, value_nn=value,
                        save_metrics_path=save_metrics_path)
    
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        agent.train(sess)


if __name__ == '__main__':
    np.random.seed(23)
    start = time.time()
    fine_tune_cartpole_to_mcc(
        save_metrics_path=config.cartpole_to_mcc_results_path,
        render=False
    )
    end = time.time()
    print("Time taken to fine tune cartpole to mountain car: ", end - start)