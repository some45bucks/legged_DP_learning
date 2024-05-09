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
    policy: Callable[..., Any],
    key: PRNGKey,
    extra_fields: Sequence[str] = ()
) -> Tuple[envs.State, Transition]:
  """Collect data."""
  actions, new_hidden_state, policy_extras = policy(env_state.obs, env_state.info['hidden_state'], key)
  env_state.info['hidden_state'] = new_hidden_state
  nstate = env.step(env_state, actions)
  state_extras = {x: nstate.info[x] for x in extra_fields}
  return nstate, Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
      observation=env_state.obs,
      hidden_state=env_state.info['hidden_state'],
      action=actions,
      reward=nstate.reward,
      discount=1 - nstate.done,
      next_observation=nstate.obs,
      next_hidden_state=nstate.info['hidden_state'],
      extras={
          'policy_extras': policy_extras,
          'state_extras': state_extras
      })


def generate_unroll(
    env: envs.Env,
    env_state: envs.State,
    policy: Callable[..., Any],
    key: PRNGKey,
    unroll_length: int,
    extra_fields: Sequence[str] = ()
) -> Tuple[envs.State, Transition]:
  """Collect trajectories of given unroll_length."""

  @jax.jit
  def f(carry, unused_t):
    state, current_key = carry
    current_key, next_key = jax.random.split(current_key)
    nstate, transition = actor_step(
        env, state, policy, current_key, extra_fields=extra_fields)
    return (nstate, next_key), transition

  (final_state, _), data = jax.lax.scan(
      f, (env_state, key), (), length=unroll_length)
  return final_state,  data
