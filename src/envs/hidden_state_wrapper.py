from typing import Callable, Dict, Optional, Tuple

from typing import Any
from brax.base import System
from brax.envs.base import Env, State, Wrapper
from flax import struct
from typing import Sequence
import jax
from jax import numpy as jp
import functools

class HiddenStateWrapper(Wrapper):

    def __init__(self, env: Env, reset_function: Callable[..., Any]):
        super().__init__(env)
        self.hidden_state_function = reset_function
        self.hidden_state = None

    def reset(self, rng: jax.Array) -> State:
        k1, k2 = jax.random.split(rng)
        state = self.env.reset(k1)
        if self.hidden_state_function != None:
            self.hidden_state = self.hidden_state_function(k2)
        else:
            self.hidden_state = None
        state.info['hidden_state'] = self.hidden_state
        return state

    def step(self, state: State, action: jax.Array) -> State:
        self.hidden_state = state.info['hidden_state']
        state = self.env.step(state, action)
        state.info['hidden_state'] = self.hidden_state
        return state