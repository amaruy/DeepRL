import os

# Path to the weights folder
WEIGHTS_PATH = '/home/etaylor/code_projects/DRL/drl_ass3/weights'

# mcc networks weights paths
mcc_policy_weights = os.path.join(WEIGHTS_PATH, 'mcc_policy.json')
mcc_value_weights = os.path.join(WEIGHTS_PATH, 'mcc_value.json')

# acrobat networks weights paths
acrobot_policy_weights = os.path.join(WEIGHTS_PATH, 'acrobat_policy.json')
acrobot_value_weights = os.path.join(WEIGHTS_PATH, 'acrobat_value.json')

# cartpole networks weights paths
cartpole_policy_weights = os.path.join(WEIGHTS_PATH, 'cartpole_policy.json')
cartpole_value_weights = os.path.join(WEIGHTS_PATH, 'cartpole_value.json')
