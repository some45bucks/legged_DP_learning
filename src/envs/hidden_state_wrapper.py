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