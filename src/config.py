import os
import datetime
# Path to the weights folder
WEIGHTS_PATH = '/home/etaylor/code_projects/DRL/drl_ass3/weights'

# path to run results
RUN_RESULTS_PATH = '/home/etaylor/code_projects/DRL/drl_ass3/results'

# datetime formatted for results run
datetime_formatted = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# -- cartpole networks configs --
# cartpole networks weights paths
cartpole_policy_weights = os.path.join(WEIGHTS_PATH, 'cartpole_policy.json')
cartpole_value_weights = os.path.join(WEIGHTS_PATH, 'cartpole_value.json')
# cartpole hidden layers
cartpole_value_hidden_1_size = 64
cartpole_value_hidden_2_size = 64
cartpole_policy_hidden_1_size = 12
cartpole_policy_hidden_2_size = 12

# acrobat networks weights paths
acrobot_policy_weights = os.path.join(WEIGHTS_PATH, 'acrobat_policy.json')
acrobot_value_weights = os.path.join(WEIGHTS_PATH, 'acrobat_value.json')
# acrobat hidden layers
acrobot_policy_hidden_layer_size = 12

# -- mcc networks configs --
# mcc networks weights paths
mcc_policy_weights = os.path.join(WEIGHTS_PATH, 'mcc_policy.json')
mcc_value_weights = os.path.join(WEIGHTS_PATH, 'mcc_value.json')
# mcc hidden layers
mcc_value_hidden_1_size = 64
mcc_value_hidden_2_size = 64
mcc_policy_hidden_1_size = 12
mcc_policy_hidden_2_size = 12

mcc_hidden_1_size = 64
mcc_hidden_2_size = 16

# all problems config
state_size = 6
action_size = 3

# cartpole actor critic config
cartpole_env_name = 'CartPole-v1'
cartpole_env_state_size = 4
cartpole_env_action_size = 2
cartpole_max_episodes = 1500
cartpole_max_steps = 501
cartpole_avg_reward_thresh = 475
cartpole_file_name = f"{datetime_formatted}_cartpole.pickle"
cartpole_run_results_path = os.path.join(RUN_RESULTS_PATH, "actor_critic/cartpole", cartpole_file_name)

# acrobot actor critic config
acrobot_env_name = 'Acrobot-v1'
acrobot_max_episodes = 750
acrobot_max_steps = 501
acrobot_avg_reward_thresh = -85
acrobot_file_name = f"{datetime_formatted}_acrobot.pickle"
acrobot_run_results_path = os.path.join(RUN_RESULTS_PATH, "actor_critic/acrobot", acrobot_file_name)

# mcc actor critic config
mcc_env_name = 'MountainCarContinuous-v0'
mcc_max_episodes = 1500
mcc_max_steps = 1000
mcc_env_state_size = 2
mcc_env_action_size = 2
mcc_actions = [-1, 1]
mcc_avg_reward_thresh = 85
mcc_file_name = f"{datetime_formatted}_mcc.pickle"
mcc_run_results_path = os.path.join(RUN_RESULTS_PATH, "actor_critic/mcc", mcc_file_name)

