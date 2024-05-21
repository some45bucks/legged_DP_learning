from functools import partial
import os
from utils.save_load import load_config, save_params, record, load_params
from train.train_ppo import train_ppo
from networks.ppo import make_ppo_network, make_ppo_policy
from rendering.display import render_rollout
from envs.custom_wrappers import CompleteAutoNormWrapper, HiddenStateWrapper
from brax.training.acme import running_statistics


from brax import envs
import envs as my_envs #don't remove this import

MAIN_FOLDER = ''
CONFIG_PATH = f"{MAIN_FOLDER}configs/debug_config.yaml"

params = load_config(CONFIG_PATH)

param_path_save = f"{MAIN_FOLDER}data/go1/params/{params['network']['name']}/params.pkl"
rollout_path_save = f"{MAIN_FOLDER}data/go1/rollouts/{params['network']['name']}/"

make_ppo_network_partial = partial(make_ppo_network,
                            head_name = params['network']['name'],        
                            head_params = params['network']['head_params'],
                            value_params = params['network']['ppo_params']['value_params'],
                            policy_params = params['network']['ppo_params']['policy_params'])

env = envs.get_environment(params['enviroment']['name'],**params['enviroment']['enviroment_params'], data_path=f"{MAIN_FOLDER}data")

if not os.path.exists(param_path_save):
    mk_policy, norm_params, policy_params, metrics = train_ppo(make_ppo_network_partial=make_ppo_network_partial,environment=env, **params['train'], data_path=f"{MAIN_FOLDER}data")

    save_params((norm_params,policy_params), path=param_path_save)

    print("Training complete")
else:
    print("Training already done")

print("Loading parameters")

ppo_net = make_ppo_network_partial(
    input = env.observation_size,
    output = env.action_size)

norm_params,policy_params = load_params(param_path_save)

env = HiddenStateWrapper(env)

env = CompleteAutoNormWrapper(env, running_statistics.normalize ,norm_params)

policy = make_ppo_policy(policy_params, ppo_net)

for rng in range(3):
    rollout = record(env, policy, rng, path=rollout_path_save)

    render_rollout(env, rollout, 1 ,title=f"Rollout {params['network']['name']} {rng}")