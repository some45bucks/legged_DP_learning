import yaml
import jax
from jax import numpy as jp
import pickle as pkl

def load_config(path="config.yaml"):
    with open(path, "r") as file:
        return yaml.safe_load(file)
    
def save_params(params, path='params.pkl'):
    with open(path, 'wb') as file:
        pkl.dump(params, file)

def load_params(path='params.pkl'):
    with open(path, 'rb') as file:
        return pkl.load(file)
    
def load_rollout(path):
    with open(path, 'rb') as file:
        return pkl.load(file)
    
def record(reset, step, policy, rng, path='/', command=None):

    key1, key2 = jax.random.split(jax.random.PRNGKey(rng))

    state = reset(key1)

    if not command is None:
        the_command = jp.array(command)
        state.info['command'] = the_command

    rollout = [state.pipeline_state]
    actions = []

    done = False

    hidden_state = state.info['hidden_state']
    foundnan = False
    while not done:
        act_rng, key2 = jax.random.split(key2)
        ctrl, hidden_state = policy(state.obs, hidden_state, act_rng)
        if jp.isnan(ctrl).any():
            print("nan in action")
            foundnan = True
            break
        actions.append(ctrl)
        state = step(state, ctrl)
        rollout.append(state.pipeline_state)
        done = state.done

    actions.append(actions[-1])

    #save the rollout
    if not foundnan:
        with open(path+f"rollout_{rng}.pkl", 'wb') as file:
            pkl.dump((rollout, actions), file)

    return rollout, actions


