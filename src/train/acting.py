import time
from typing import Callable, Sequence, Tuple, Union, Any, NamedTuple

from brax import envs
from brax.training.types import PRNGKey
import jax
from jax import numpy as jp

class Transition(NamedTuple):
  hidden_state: jp.ndarray
  observation: jp.ndarray
  action: jp.ndarray
  reward: jp.ndarray
  discount: jp.ndarray
  next_observation: jp.ndarray
  next_hidden_state: jp.ndarray
  extras: jp.ndarray = () 


def actor_step(
    env: envs.Env,
    env_state: envs.State,
    hidden_state: jp.ndarray,
    policy: Callable[..., Any],
    key: PRNGKey,
    extra_fields: Sequence[str] = ()
) -> Tuple[envs.State, Transition]:
  """Collect data."""
  actions, new_hidden_state ,policy_extras = policy(env_state.obs, hidden_state, key)
  nstate = env.step(env_state, actions)
  state_extras = {x: nstate.info[x] for x in extra_fields}
  return nstate, new_hidden_state, Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
      observation=env_state.obs,
      hidden_state=hidden_state,
      action=actions,
      reward=nstate.reward,
      discount=1 - nstate.done,
      next_observation=nstate.obs,
      next_hidden_state=new_hidden_state,
      extras={
          'policy_extras': policy_extras,
          'state_extras': state_extras
      })


def generate_unroll(
    env: envs.Env,
    env_state: envs.State,
    starting_hidden_state: jp.ndarray,
    policy: Callable[..., Any],
    key: PRNGKey,
    unroll_length: int,
    extra_fields: Sequence[str] = ()
) -> Tuple[envs.State, Transition]:
  """Collect trajectories of given unroll_length."""

  @jax.jit
  def f(carry, unused_t):
    state, hidden_state, current_key = carry
    current_key, next_key = jax.random.split(current_key)
    nstate, new_hidden_state, transition = actor_step(
        env, state, hidden_state, policy, current_key, extra_fields=extra_fields)
    return (nstate, new_hidden_state, next_key), transition

  (final_state, new_hidden_state, _), data = jax.lax.scan(
      f, (env_state, starting_hidden_state, key), (), length=unroll_length)
  return final_state, new_hidden_state,  data
