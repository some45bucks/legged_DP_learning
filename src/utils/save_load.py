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
    
def record(env, policy, rng, render, path='rollout.pkl', command=None):
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    jit_inference_fn = jax.jit(policy)

    state = jit_reset(rng)

    if not command is None:
        the_command = jp.array(command)
        state.info['command'] = the_command

    rollout = [state.pipeline_state]

    done = False

    while not done:
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, ctrl)
        rollout.append(state.pipeline_state)

        if render:
            env.render()

        done = state.done

    #save the rollout
    with open(path, 'wb') as file:
        pkl.dump(rollout, file)

    return rollout


