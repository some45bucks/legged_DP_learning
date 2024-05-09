from typing import Callable, Dict, Optional, Tuple

from brax.base import System
from brax.envs.base import Env, State, Wrapper
from flax import struct
from typing import Sequence
import jax
from jax import numpy as jp
import functools


def starting_hidden_state(state_size: int) -> jp.ndarray:
    return jp.zeros(state_size)

class HiddenStateWrapper(Wrapper):

    def __init__(self, env: Env, state_size: int):
        super().__init__(env)
        self.hidden_state_function = functools.partial(starting_hidden_state, state_size)
        self.hidden_state = None

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        self.hidden_state = self.hidden_state_function()
        state.info['hidden_state'] = self.hidden_state
        return state

    def step(self, state: State, action: jax.Array) -> State:
        self.hidden_state = state.info['hidden_state']
        state = self.env.step(state, action)
        state.info['hidden_state'] = self.hidden_state
        return state