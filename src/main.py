from functools import partial
import os
from utils.save_load import load_config, save_params, record, load_params
from train.train_ppo import train_ppo
from train.train_env import train_env
from networks.ppo import make_ppo_network, make_ppo_policy
from networks.networks import make_feed_forward, make_lstm
from rendering.display import render_rollout, pretty_print_object
from envs.custom_wrappers import CompleteAutoNormWrapper, HiddenStateWrapper, NetWrapper
from brax.training.acme import running_statistics
from utils.data_funcs import gather_rollout_data


from brax import envs
import jax
import envs as _envs #don't remove this import

MAIN_FOLDER = '/home/jakehate/phys_sim/'
CONFIG_PATH = f"{MAIN_FOLDER}configs/debug_config.yaml"
params = load_config(CONFIG_PATH)
save_folder = f"{MAIN_FOLDER}data/go1/{params['agent_network']['name']}"
network_param_path_save = f"{save_folder}/network_params/"
env_param_path_save = f"{save_folder}/env_params/"
rollout_path_save = f"{save_folder}/rollouts/"

os.makedirs(network_param_path_save, exist_ok=True)
os.makedirs(rollout_path_save, exist_ok=True)
os.makedirs(env_param_path_save, exist_ok=True)

ROLL_OUTS = 50

env = envs.get_environment(params['enviroment']['name'],**params['enviroment']['enviroment_params'], scene_path=f"{MAIN_FOLDER}data/go1/")

make_in_network_partial = partial(make_feed_forward,**params['env_network']['in_params'])
make_out_network_partial = partial(make_feed_forward,**params['env_network']['out_params'])

final_env_param_save = env_param_path_save+"params.pkl"

if not os.path.exists(final_env_param_save) and os.path.exists(rollout_path_save+f"rollout_{ROLL_OUTS}.pkl"):
    print('Starting training Env...')
    
    train_data = gather_rollout_data(rollout_path_save, ROLL_OUTS+1)

    norm_params, env_network_params, type_params = train_env(train_data=train_data,make_in_part=make_in_network_partial, make_out_part=make_out_network_partial,environment=env, **params['env_train'], param_path=env_param_path_save)

    save_params((norm_params,env_network_params, type_params), path=final_env_param_save)

    print("Training complete")
else:
    print("Training already done/ Not enough rollouts")

make_ppo_network_partial = partial(make_ppo_network,
                            head_name = params['agent_network']['name'],        
                            head_params = params['agent_network']['head_params'],
                            value_params = params['agent_network']['ppo_params']['value_params'],
                            policy_params = params['agent_network']['ppo_params']['policy_params'])

final_param_save = network_param_path_save+"params.pkl"

if not os.path.exists(final_param_save):
    print('Starting training PPO agent...')

    if os.path.exists(final_env_param_save):
        print('Loading env params')
        env_params = load_params(final_env_param_save)
    else:
        print('No env params found')
        env_params = None
            
    mk_policy, norm_params, policy_params, wrap_params ,metrics = train_ppo(make_ppo_network_partial=make_ppo_network_partial,make_in_part=make_in_network_partial,make_out_part=make_out_network_partial,environment=env, **params['agent_train'], param_path=network_param_path_save, env_params=env_params)

    save_params((norm_params,policy_params), path=final_param_save)

    print("Training complete")
else:
    print("Training already done")

from jax import numpy as jp
import functools

if not os.path.exists(rollout_path_save+f"rollout_{ROLL_OUTS}.pkl"):

    print("Loading parameters")

    ppo_net = make_ppo_network_partial(
        input = env.observation_size,
        output = env.action_size)

    norm_params,policy_params = load_params(final_param_save)

    

    if os.path.exists(final_env_param_save):
        print('Loading env params')
        env_params = load_params(final_env_param_save)
        types = jp.stack(env_params[2]['params']['0'], axis=0)
        type_mean = jp.mean(types, axis=0)
        type_cov =  jp.cov(types.T)

        type_dist_fn = functools.partial(jax.random.multivariate_normal, mean=type_mean, cov=type_cov)

        test_type = type_dist_fn(jax.random.PRNGKey(0))

        assert float('nan') != test_type[0], 'not enough data for a cov matrix'

        in_net = make_in_network_partial(
            input_size = env.action_size + types.size,
            output_size = env.action_size)
        
        out_net = make_out_network_partial(
            input_size = env.vel_pos + types.size,
            output_size = env.vel_pos)
        
        gen_func = jax.jit(env.pipeline_init)
        
        wrap_params = (in_net, out_net, env_params[1][0], env_params[1][1], env_params[0], type_dist_fn, gen_func)
        print('Env params loaded')
    else:
        print('No env params found')
        wrap_params = None

    env = HiddenStateWrapper(env)

    if wrap_params is not None:
        env = NetWrapper(env, *wrap_params)

    env = CompleteAutoNormWrapper(env, running_statistics.normalize ,norm_params)

    policy = make_ppo_policy(policy_params, ppo_net)
    
    j_reset = jax.jit(env.reset)
    j_step = jax.jit(env.step)
    j_policy = jax.jit(policy)

    for rng in range(ROLL_OUTS+1):
        print(f"Rollout {rng}")
        data = record(j_reset, j_step, j_policy, rng, path=rollout_path_save)

        if rng % (max(ROLL_OUTS//5,1)) == 0:
            render_rollout(env, data[0], 1 ,title=f"Rollout {params['agent_network']['name']} {rng}")

else:
    print("Rollouts already done")