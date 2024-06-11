import time
from typing import Callable, Sequence, Tuple, Union, Any, NamedTuple, Optional
import functools
from brax import envs
from brax.training.types import PRNGKey
import jax
from jax import numpy as jp
from brax.training.acme import running_statistics

from networks.ppo import ppo_network, ppo_network_params
from utils.data_funcs import extract_q_dq

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

def unroll_policy(ppo_network: ppo_network,
          normalizer_params: Any,
          params: ppo_network_params,
          start_state: envs.State,
          rng: jp.ndarray,
          env: envs.Env,
          unroll_length: int,
          env_params: Optional[Tuple] = None,
          gen_func: Optional[Callable] = None,
          net1: Optional[Callable] = None,
          net2: Optional[Callable] = None
          ) -> Tuple[Any,Transition]:
      
      assert gen_func is None and net1 is None and net2 is None if env_params is None else gen_func is not None and net1 is not None and net2 is not None, f"Must all be the same. env_params: {env_params}, gen_func: {gen_func}, net1: {net1}, net2: {net2}"

      @jax.jit
      def step(carry: Tuple[envs.State, jp.ndarray], unused_t) -> Tuple[Any,Transition]:
          state, current_key = carry

          current_key, next_key = jax.random.split(current_key)

          hidden_state = state.info['hidden_state']

          hidden, next_hidden_state = ppo_network.head_network.apply(params.head, state.obs, hidden_state)

          policy_logits, _ = ppo_network.policy_network.apply(params.policy, hidden, None)

          raw_actions = ppo_network.action_distribution.sample_no_postprocessing(policy_logits, current_key)
                
          log_prob = ppo_network.action_distribution.log_prob(policy_logits, raw_actions)

          actions = ppo_network.action_distribution.postprocess(raw_actions)

          baseline, _ = ppo_network.value_network.apply(params.value, hidden, None)

          baseline = jp.squeeze(baseline, axis=-1)

          def network_fn(state):
                input = extract_q_dq([state.pipeline_state])[0]
                concat_input =  jp.concatenate((input[0], input[1]),1)

                n_concat_input = running_statistics.normalize(concat_input, env_params[0])

                n_concat_input =  jp.concatenate((n_concat_input, state.info['env_type']),1)

                output, _ = net2.apply(env_params[1][1],n_concat_input, None)
                new_pipe = gen_func(output[:,:19],output[:,19:])
                state = state.replace(pipeline_state=new_pipe)
                return state

          if env_params is None:
            next_state = env.step(state, actions, normalizer_params)
          else:
            concat_actions =  jp.concatenate((actions, state.info['env_type']),1)

            net_action, _ = net1.apply(env_params[1][0],concat_actions,  None)

            new_state = env.step(state,net_action,normalizer_params)

            next_state = network_fn(new_state)               

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

      data = jax.tree_util.tree_map(lambda x: jp.swapaxes(x, 0, 1), data)

      return final_state, data
