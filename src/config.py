import os
import datetime

# Ensure directories exist
def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Base paths
WEIGHTS_PATH = 'weights'
RUN_RESULTS_PATH = 'results'
LOGS_PATH = 'logs'

ensure_directory_exists(WEIGHTS_PATH)
ensure_directory_exists(RUN_RESULTS_PATH)
ensure_directory_exists(LOGS_PATH)

# Current time for file naming
datetime_formatted = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# Cartpole configurations
cartpole_policy_weights = os.path.join(WEIGHTS_PATH, 'cartpole_policy.json')
cartpole_value_weights = os.path.join(WEIGHTS_PATH, 'cartpole_value.json')
cartpole_value_hidden_1_size = 64
cartpole_value_hidden_2_size = 64
cartpole_policy_hidden_1_size = 12
cartpole_policy_hidden_2_size = 12
cartpole_prog_value_hidden_size = 64
cartpole_prog_policy_hidden_size = 12
cartpole_env_name = 'CartPole-v1'
cartpole_env_state_size = 4
cartpole_env_action_size = 2
cartpole_max_episodes = 1500
cartpole_max_steps = 501
cartpole_avg_reward_thresh = 475
cartpole_file_name = f"{datetime_formatted}_cartpole.pickle"
cartpole_run_results_path = os.path.join(RUN_RESULTS_PATH, "actor_critic/cartpole", cartpole_file_name)
ensure_directory_exists(os.path.dirname(cartpole_run_results_path))
cartpole_log_dir = os.path.join(LOGS_PATH, "actor_critic/cartpole")

# Acrobot configurations
acrobot_policy_weights = os.path.join(WEIGHTS_PATH, 'acrobot_policy.json')
acrobot_value_weights = os.path.join(WEIGHTS_PATH, 'acrobot_value.json')
acrobot_policy_hidden_layer_size = 12
acrobot_value_hidden_1_size = 64
acrobot_value_hidden_2_size = 16
acrobot_env_name = 'Acrobot-v1'
acrobot_env_action_size = 3
acrobot_max_episodes = 750
acrobot_max_steps = 501
acrobot_avg_reward_thresh = -85
acrobot_file_name = f"{datetime_formatted}_acrobot.pickle"
acrobot_run_results_path = os.path.join(RUN_RESULTS_PATH, "actor_critic/acrobot", acrobot_file_name)
ensure_directory_exists(os.path.dirname(acrobot_run_results_path))
acrobot_log_dir = os.path.join(LOGS_PATH, "actor_critic/acrobot")

# MCC configurations
mcc_policy_weights = os.path.join(WEIGHTS_PATH, 'mcc_policy.json')
mcc_value_weights = os.path.join(WEIGHTS_PATH, 'mcc_value.json')
mcc_value_hidden_1_size = 64
mcc_value_hidden_2_size = 64
mcc_policy_hidden_1_size = 12
mcc_policy_hidden_2_size = 12
mcc_prog_value_hidden_size = 32
mcc_prog_policy_hidden_size = 12
mcc_env_name = 'MountainCarContinuous-v0'
mcc_max_episodes = 1500
mcc_max_steps = 1000
mcc_env_state_size = 2
mcc_env_action_size = 2
mcc_actions = [-1, 1]
mcc_avg_reward_thresh = 85
mcc_file_name = f"{datetime_formatted}_mcc.pickle"
mcc_run_results_path = os.path.join(RUN_RESULTS_PATH, "actor_critic/mcc", mcc_file_name)
ensure_directory_exists(os.path.dirname(mcc_run_results_path))
mcc_log_dir = os.path.join(LOGS_PATH, "actor_critic/mcc")

# Fine tune and transfer learning configurations
acrobot_to_cartpole_file_name = f"{datetime_formatted}_acrobot_to_cartpole.pickle"
acrobot_to_cartpole_results_path = os.path.join(RUN_RESULTS_PATH, "fine_tune/acrobot_to_cartpole", acrobot_to_cartpole_file_name)
ensure_directory_exists(os.path.dirname(acrobot_to_cartpole_results_path))
acrobot_to_cartpole_log_dir = os.path.join(LOGS_PATH, "fine_tune/acrobot_to_cartpole")

cartpole_to_mcc_file_name = f"{datetime_formatted}_cartpole_to_mcc.pickle"
cartpole_to_mcc_results_path = os.path.join(RUN_RESULTS_PATH, "fine_tune/cartpole_to_mcc", cartpole_to_mcc_file_name)
ensure_directory_exists(os.path.dirname(cartpole_to_mcc_results_path))
cartpole_to_mcc_log_dir = os.path.join(LOGS_PATH, "fine_tune/cartpole_to_mcc")

prog_cartpole_file_name = f"{datetime_formatted}_prog_cartpole.pickle"
prog_cartpole_run_results_path = os.path.join(RUN_RESULTS_PATH, "transfer_learning/prog_cartpole", prog_cartpole_file_name)
ensure_directory_exists(os.path.dirname(prog_cartpole_run_results_path))
prog_cartpole_log_dir = os.path.join(LOGS_PATH, "transfer_learning/prog_cartpole")

prog_mcc_file_name = f"{datetime_formatted}_prog_mcc.pickle"
prog_mcc_run_results_path = os.path.join(RUN_RESULTS_PATH, "transfer_learning/prog_mcc", prog_mcc_file_name)
ensure_directory_exists(os.path.dirname(prog_mcc_run_results_path))
prog_mcc_log_dir = os.path.join(LOGS_PATH, "transfer_learning/prog_mcc")

# All problems config
state_size = 6
action_size = 3

# Ensure base directories for weights and logs are created
ensure_directory_exists(WEIGHTS_PATH)
ensure_directory_exists(RUN_RESULTS_PATH)
ensure_directory_exists(LOGS_PATH)
