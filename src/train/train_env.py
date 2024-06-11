import jax
from jax import numpy as jp
import numpy as np
from typing import Tuple, Union

import functools
import time
from typing import Callable, Optional, Tuple, Union, Any, Sequence

from absl import logging
from brax import base
from brax import envs
from train import acting
from brax.training import pmap
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.types import Params
from brax.training.types import PRNGKey
import flax
import jax
import numpy as np
import optax

from train.evaluator import evaluator
from train.gradients import gradient_update_fn as gradient_update
from train.losses import compute_env_loss, compute_env_loss_type
from envs.custom_wrappers import HiddenStateWrapper, AutoNormWrapper
from rendering.display import get_progress_fn
from utils.save_load import save_params
from networks.networks import Network

from utils.data_funcs import data_sequence

InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]
Metrics = types.Metrics

_PMAP_AXIS_NAME = 'i'

@flax.struct.dataclass
class TrainingState:
  """Contains training state for the learner."""
  net_optimizer_state: optax.OptState
  type_optimizer_state: optax.OptState
  params: Tuple[jp.ndarray, jp.ndarray]
  full_type_params: Sequence[jp.ndarray]
  normalizer_params: Tuple[running_statistics.RunningStatisticsState, running_statistics.RunningStatisticsState]
  env_steps: jp.ndarray

def _unpmap(v):
  return jax.tree_util.tree_map(lambda x: x[0], v)


def _strip_weak_type(tree):
  # brax user code is sometimes ambiguous about weak_type.  in order to
  # avoid extra jit recompilations we strip all weak types from user input
  def f(leaf):
    leaf = jp.asarray(leaf)
    return leaf.astype(leaf.dtype)
  return jax.tree_util.tree_map(f, tree)


def wrap(env, normalize=None):
  env = envs.training.VmapWrapper(env)
  # if normalize != None:
  #   env = AutoNormWrapper(env,normalize)
  
  return env


def save_ppo_params(steps ,params: Params,name, param_path):
  print(f'Saving params... name:{name} steps:{steps}')
  save_params(params,f'{param_path}_ppo_params_{steps}.pkl')

def train_env(
    train_data: Sequence[jp.ndarray],
    make_in_part: Callable[..., Network],
    make_out_part: Callable[..., Network],  
    environment: envs.Env,
    name: str = 'Default',
    learning_rate: float = 1e-4,
    seed: int = 0,
    unroll_length: int = 10,
    type_size: int = 4,
    type_split_every: int = 20, 
    data_loops: int = 100,
    num_minibatches: int = 16,
    normalize_observations: bool = True,
    progress_fn: Callable[[int, Metrics], None] = get_progress_fn(),
    policy_params_fn: Callable[..., None] = save_ppo_params,
    randomization_fn: Optional[Callable[[base.System, jp.ndarray], Tuple[base.System, base.System]]] = None,
    param_path: str = 'data/go1/default/params',
):
  
  train_data = data_sequence(type_split_every,unroll_length,type_size,train_data)

  data_length = len(train_data)

  xt = time.time()

  process_count = jax.process_count()
  process_id = jax.process_index()
  local_device_count = jax.local_device_count()
  local_devices_to_use = local_device_count

  assert data_length % (num_minibatches // process_count) == 0, f"Data length ({data_length}) must be divisible by num_minibatches ({num_minibatches}) // process_count ({process_count})"
  
  logging.info(
      'Device count: %d, process count: %d (id %d), local device count: %d, '
      'devices to be used count: %d', jax.device_count(), process_count,
      process_id, local_device_count, local_devices_to_use)
  device_count = local_devices_to_use * process_count

  key = jax.random.PRNGKey(seed)
  global_key, local_key = jax.random.split(key)
  del key
  local_key = jax.random.fold_in(local_key, process_id)
  local_key, key_env = jax.random.split(local_key, 2)
  # key_networks should be global, so that networks are initialized the same
  # way for different processes.
  key_in, key_out, key_type = jax.random.split(global_key,3)

  del global_key

  assert num_minibatches % device_count == 0

  v_randomization_fn = None
  if randomization_fn is not None:
    randomization_batch_size = num_minibatches // local_device_count
    # all devices gets the same randomization rng
    randomization_rng = jax.random.split(key_env, randomization_batch_size)
    v_randomization_fn = functools.partial(
        randomization_fn, rng=randomization_rng
    )

  normalize = lambda x, y: x
  if normalize_observations:
    normalize = running_statistics.normalize

  env = wrap(environment, normalize)
  reset_fn = jax.jit(jax.vmap(env.reset))
  key_envs = jax.random.split(key_env, num_minibatches // process_count)
  key_envs = jp.reshape(key_envs,
                         (local_devices_to_use, -1) + key_envs.shape[1:])
  
  reset_state = reset_fn(key_envs)

  net_optimizer = optax.adam(learning_rate=learning_rate)
  type_optimizer = optax.adam(learning_rate=learning_rate)

  in_net = make_in_part(
      input_size = environment.action_size + type_size,
      output_size = environment.action_size)
  
  out_net = make_out_part(
      input_size = environment.vel_pos + type_size,
      output_size = environment.vel_pos)
  
  main_slice = [-1,-1]

  loss_fn = functools.partial(
    compute_env_loss,
    network=(in_net, out_net),
    env=env,
    reset_state = reset_state,
    slice = main_slice
    )
  
  loss_fn_type = functools.partial(
    compute_env_loss_type,
    network=(in_net, out_net),
    env=env,
    reset_state = reset_state,
    slice = main_slice
    )

  net_gradient_update_fn = gradient_update(loss_fn, net_optimizer, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True)
  type_gradient_update_fn = gradient_update(loss_fn_type, type_optimizer, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True)
  
  def step(net_optimizer_state, type_optimizer_state, net_params, type_params, normalizer_params, data_chunk, key):
    key, key_loss= jax.random.split(key)

    print("starting step compile...")
    (_, out_data1), net_params_out, net_optimizer_state = net_gradient_update_fn(
        net_params,
        type_params,
        data_chunk,
        normalizer_params,
        key_loss,
        optimizer_state=net_optimizer_state)

    (_, out_data2), full_type_params, type_optimizer_state = type_gradient_update_fn(
        type_params,
        net_params,
        data_chunk,
        normalizer_params,
        key_loss,
        optimizer_state=type_optimizer_state)
    
    metrics = (out_data1['loss_metrics'], out_data2['loss_metrics'])

    return net_optimizer_state, type_optimizer_state, net_params_out, full_type_params, out_data2['normalizer_params'], metrics

  def training_epoch(training_state: TrainingState, data_chunk, key: PRNGKey) -> Tuple[TrainingState, Any]:
    
    print("starting training epoch compile...")

    net_optimizer_state, type_optimizzer_state, net_params, full_type_params, norm_params, loss_metrics = step(training_state.net_optimizer_state,
                                                                                                                    training_state.type_optimizer_state,
                                                                                                                    training_state.params,
                                                                                                                    training_state.full_type_params,
                                                                                                                    training_state.normalizer_params,
                                                                                                                    data_chunk,
                                                                                                                    key)
    print("finished step compile!")

    new_training_state = TrainingState(
      net_optimizer_state=net_optimizer_state,
      type_optimizer_state=type_optimizzer_state,
      params=net_params,
      full_type_params=full_type_params,
      normalizer_params=norm_params,
      env_steps=0)

    loss_metrics = jax.tree_util.tree_map(jp.mean, loss_metrics)

    return new_training_state, loss_metrics

  training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)
  
  net_params = (
    in_net.init(key_in),
    out_net.init(key_out),
  )

  type_params = train_data.get_params(key_type)

  training_state = TrainingState(
      net_optimizer_state=net_optimizer.init(net_params),
      type_optimizer_state=type_optimizer.init(type_params),
      params=net_params,
      full_type_params=type_params,
      normalizer_params=(running_statistics.init_state(
          specs.Array(environment.vel_pos, jp.dtype('float32')))),
      env_steps=0)
  
  training_state = jax.device_put_replicated(
      training_state,
      jax.local_devices()[:local_devices_to_use])

  for it in range(data_loops):
    print(f'Iteration {it}')
    logging.info('starting iteration %s %s', it, time.time() - xt)
    av_network_loss = 0
    av_type_loss = 0
    for i in range(0,data_length, (num_minibatches // process_count)):

      main_slice[0] = i
      main_slice[1] = i + (num_minibatches // process_count)

      data_chunk = train_data[main_slice[0]:main_slice[1]]

      epoch_key, local_key = jax.random.split(local_key)
      epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
      
      training_state, data_chunk = _strip_weak_type((training_state, data_chunk))
      result = training_epoch(training_state, data_chunk ,epoch_keys)

      if main_slice[0] == 0 and it == 0:
        print("finished training epoch compile!")

      training_state, metrics = _strip_weak_type(result)
      training_metrics = jax.tree_util.tree_map(jp.mean, metrics)
      jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

      av_network_loss += training_metrics[0]['total_loss']
      av_type_loss += training_metrics[1]['total_loss']

      key_envs = jax.vmap(
          lambda x, s: jax.random.split(x[0], s),
          in_axes=(0, None))(key_envs, key_envs.shape[1])
      
    av_network_loss /= data_length
    av_type_loss /= data_length
    print(f'Average network loss: {av_network_loss}, Average type loss: {av_type_loss}')

  # If there was no mistakes the training_state should still be identical on all
  # devices.
  pmap.assert_is_replicated(training_state)
  pmap.synchronize_hosts()
  return _unpmap(training_state.normalizer_params), _unpmap(training_state.params), _unpmap(training_state.full_type_params)
