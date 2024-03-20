import tensorflow as tf
from src.networks.acrobot.acrobot_policy_network import AcrobotPolicyNetwork
from src.networks.acrobot.acrobot_value_network import AcrobotValueNetwork
from src.actor_critic.cartpole import CartpoleActorCritic
import time
from src import config
import numpy as np

def fine_tune_acrobot_to_cartpole(save_metrics_path=None, value_lr=0.0008, policy_lr=0.0005, discount_factor=0.99, 
                               restore_weights=True, render=True):
    """
    Fine Tune acrobot nets to  cartpole data actor-critic agent.

    Parameters:
    - save_metrics_path: Path to save the metrics of the training.
    - value_lr: Learning rate for the value network.
    - policy_lr: Learning rate for the policy network.
    - discount_factor: Discount factor for future rewards.
    - restore_weights: Whether to restore weights for the networks.
    - render: Whether to render the environment.
    """
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()

    # Load the weights for the acrobots value and policy networks
    value = AcrobotValueNetwork(value_lr, restore_weights=restore_weights)
    policy = AcrobotPolicyNetwork(config.cartpole_env_action_size, policy_lr, restore_weights=restore_weights)
    
    # Initialize actor critic agent on the Cartpole environment
    agent = CartpoleActorCritic(discount_factor, policy_lr, value_lr, render=render, 
                                policy_nn=policy,
                                value_nn=value,
                                save_metrics_path=save_metrics_path)
    
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        agent.train(sess)

if __name__ == '__main__':
    np.random.seed(23)
    start = time.time()
    fine_tune_acrobot_to_cartpole(
        save_metrics_path=config.acrobot_to_cartpole_results_path,
        render=False
    )
    end = time.time()
    print("Time taken to fine tune acrobot to cartpole: ", end - start)
