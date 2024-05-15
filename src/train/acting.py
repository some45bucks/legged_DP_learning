import time
from typing import Callable, Sequence, Tuple, Union, Any, NamedTuple

from brax import envs
from brax.training.types import PRNGKey
import jax
from jax import numpy as jp

from networks.ppo import ppo_network, ppo_network_params

class Transition(NamedTuple):
  hidden_state: jp.ndarray
  observation: jp.ndarray
  logits: jp.ndarray
  raw_actions: jp.ndarray
  log_prob: jp.ndarray
  baseline: jp.ndarray
  action: jp.ndarray
  reward: jp.ndarray
  discount: jp.ndarray
  next_observation: jp.ndarray
  next_hidden_state: jp.ndarray
  truncation: jp.ndarray

def unroll(ppo_network: ppo_network,
           normalizer_params: Any,
           params: ppo_network_params,
           start_state: envs.State,
           rng: jp.ndarray,
           env: envs.Env,
           unroll_length: int,
           ) -> Tuple[Any,Transition]:
      
      @jax.jit
      def step(carry: Tuple[envs.State, Any, jp.ndarray], unused_t) -> Tuple[Any,Transition]:
          state, current_key = carry

          current_key, next_key = jax.random.split(current_key)

          hidden_state = state.info['hidden_state']

          observations = ppo_network.normalizer(state.obs, normalizer_params)

          hidden, next_hidden_state = ppo_network.head_network.apply(params.head, observations, hidden_state)

          policy_logits, _ = ppo_network.policy_network.apply(params.policy, hidden, None)

          raw_actions = ppo_network.action_distribution.sample_no_postprocessing(policy_logits, current_key)
                
          log_prob = ppo_network.action_distribution.log_prob(policy_logits, raw_actions)

          actions = ppo_network.action_distribution.postprocess(raw_actions)

          baseline, _ = ppo_network.value_network.apply(params.value, hidden, None)

          baseline = jp.squeeze(baseline, axis=-1)

          next_state = env.step(state, actions)              

          return (next_state, next_key), Transition( 
            observation=state.obs,
            raw_actions=raw_actions,
            log_prob=log_prob,
            baseline=baseline,
            hidden_state=hidden_state,
            logits=policy_logits,
            action=actions,
            reward=next_state.reward,
            discount=1 - next_state.done,
            next_observation=next_state.obs,
            next_hidden_state=next_hidden_state,
            truncation=next_state.info['truncation']
            )

      (final_state, rng), data = jax.lax.scan(step, (start_state, rng), (), length=unroll_length)

      def conditional_reshape(x):
        if x.ndim == 2:
            return x[:, :, jp.newaxis]
        return x

      # data = jax.tree_util.tree_map(conditional_reshape, data)
      # data = jax.tree_util.tree_map(lambda x: jp.swapaxes(x, 1, 2), data)
      # data = jax.tree_util.tree_map(lambda x: jp.reshape(x, (-1,) + x.shape[2:]),
      #                               data)
      
      # data = jax.tree_util.tree_map(convert_data, data)

      data = jax.tree_util.tree_map(lambda x: jp.swapaxes(x, 0, 1), data)

      return final_state, data
