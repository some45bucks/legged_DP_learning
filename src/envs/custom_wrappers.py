from typing import Callable, Dict, Optional, Tuple

from typing import Any
from brax.base import System
from brax.envs.base import Env, State, Wrapper
from flax import struct
from typing import Sequence
import jax
from jax import numpy as jp
import functools
from utils.data_funcs import extract_q_dq
from brax.training.acme import running_statistics

class HiddenStateWrapper(Wrapper):

    def __init__(self, env: Env):
        super().__init__(env)
        self.hidden_state = None

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info['hidden_state'] = None
        return state

    def step(self, state: State, action: jax.Array) -> State:
        self.hidden_state = state.info['hidden_state']
        state = self.env.step(state, action)
        state.info['hidden_state'] = self.hidden_state
        return state
    
class AutoNormWrapper(Wrapper):

    def __init__(self, env: Env, norm):
        super().__init__(env)
        self.norm = norm

    def reset(self, rng: jax.Array, params) -> State:
        state = self.env.reset(rng)
        # if params is not None:
        #     obs = self.norm(state.obs, params)
        #     state = state.replace(obs=obs)
        return state

    def step(self, state: State, action: jax.Array, params) -> State:
        state = self.env.step(state, action)
        # obs = self.norm(state.obs, params)
        # state = state.replace(obs=obs)
        return state
    
class CompleteAutoNormWrapper(Wrapper):

    def __init__(self, env: Env, norm, params):
        super().__init__(env)
        self.norm = norm
        self.params = params

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        # obs = self.norm(state.obs, self.params)
        # state = state.replace(obs=obs)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        state = self.env.step(state, action)
        # obs = self.norm(state.obs, self.params)
        # state = state.replace(obs=obs)
        return state
    
class NetWrapper(Wrapper):

    def __init__(self, env: Env, net1, net2, net1_params, net2_params, normalizer_params, dist_fn, gen_func):
        super().__init__(env)
        self.net1 = net1
        self.net2 = net2
        self.net1_params = net1_params
        self.net2_params = net2_params
        self.normalizer_params = normalizer_params
        self.dist_fn = dist_fn
        self.gen_func = gen_func

    def network_in_fn(self, state, action):

        concat_actions =  jp.concatenate((action, state.info['env_type']),-1)

        net_action, _ = self.net1.apply(self.net1_params,concat_actions,  None)

        return net_action 

    def network_out_fn(self, new_state):

        input = extract_q_dq([new_state.pipeline_state])[0]

        concat_input =  jp.concatenate((input[0], input[1]),-1)

        n_concat_input = running_statistics.normalize(concat_input, self.normalizer_params)

        n_concat_input =  jp.concatenate((n_concat_input, new_state.info['env_type']),-1)

        output, _ = self.net2.apply(self.net2_params,n_concat_input, None)

        new_pipe = self.gen_func(output[:19],output[19:])

        new_state = new_state.replace(pipeline_state=new_pipe)

        return new_state
    
    def reset(self, rng: jax.Array) -> State:

        rng1, rng2 = jax.random.split(rng)

        new_state = self.env.reset(rng1)

        new_state.info['env_type'] = self.dist_fn(rng2)

        new_state = self.network_out_fn(new_state)

        return new_state

    def step(self, state: State, action: jax.Array) -> State:
        net_action = self.network_in_fn(state, action)
        new_state = self.env.step(state, net_action)
        new_state = self.network_out_fn(new_state)
        return new_state
