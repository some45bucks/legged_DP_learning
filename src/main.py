from functools import partial

from utils.save_load import load_config
from train.train_ppo import train_ppo
from networks.ppo import make_ppo_network

from brax import envs
import envs as my_envs

CONFIG_PATH = "configs/debug_config.yaml"

def main():
    params = load_config(CONFIG_PATH)

    make_ppo_network_partial = partial(make_ppo_network,
                               head_name = params['network']['name'],        
                               head_params = params['network']['head_params'],
                               value_params = params['network']['ppo_params']['value_params'],
                               policy_params = params['network']['ppo_params']['policy_params'])
    
    env = envs.get_environment(params['enviroment']['name'],**params['enviroment']['enviroment_params'])

    train_ppo(make_ppo_network_partial=make_ppo_network_partial,environment=env, **params['train'])


if __name__ == "__main__":
    main()