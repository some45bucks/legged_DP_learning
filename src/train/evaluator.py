import time
from typing import Callable, Any

from brax import envs
from brax.training.types import PRNGKey
import jax
from jax import numpy as jp

from train.acting import generate_unroll
from networks.ppo import infrence_fn

class evaluator:
  """Class to run evaluations."""

  def __init__(self, eval_env: envs.Env,
               eval_policy_fn: infrence_fn, num_eval_envs: int,
               episode_length: int, action_repeat: int, key: PRNGKey):
    self._key = key
    self._eval_walltime = 0

    eval_env = envs.training.EvalWrapper(eval_env)

    def generate_eval_unroll(params: Any, key: PRNGKey) -> envs.State:
      reset_keys = jax.random.split(key, num_eval_envs)
      eval_first_state = eval_env.reset(reset_keys)
      return generate_unroll(
          eval_env,
          eval_first_state,
          eval_policy_fn.starting_hidden_state(num_eval_envs),
          eval_policy_fn(params),
          key,
          unroll_length=episode_length // action_repeat)[0]

    self._generate_eval_unroll = jax.jit(generate_eval_unroll)
    self._steps_per_unroll = episode_length * num_eval_envs

  def run_evaluation(self,
                     params: Any,
                     training_metrics: Any,
                     aggregate_episodes: bool = True) -> Any:
    """Run one epoch of evaluation."""
    self._key, unroll_key = jax.random.split(self._key)

    t = time.time()
    eval_state = self._generate_eval_unroll(params, unroll_key)
    eval_metrics = eval_state.info['eval_metrics']
    eval_metrics.active_episodes.block_until_ready()
    epoch_eval_time = time.time() - t
    metrics = {}
    for fn in [jp.mean, jp.std]:
      suffix = '_std' if fn == jp.std else ''
      metrics.update(
          {
              f'eval/episode_{name}{suffix}': (
                  fn(value) if aggregate_episodes else value
              )
              for name, value in eval_metrics.episode_metrics.items()
          }
      )
    metrics['eval/avg_episode_length'] = jp.mean(eval_metrics.episode_steps)
    metrics['eval/epoch_eval_time'] = epoch_eval_time
    metrics['eval/sps'] = self._steps_per_unroll / epoch_eval_time
    self._eval_walltime = self._eval_walltime + epoch_eval_time
    metrics = {
        'eval/walltime': self._eval_walltime,
        **training_metrics,
        **metrics
    }

    return metrics  # pytype: disable=bad-return-type  # jax-ndarray