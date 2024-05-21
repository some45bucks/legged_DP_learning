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
from train.losses import compute_env_loss, compute_type_loss
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
  type_params: Sequence[jp.ndarray]
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
  if normalize != None:
    env = AutoNormWrapper(env,normalize)
  
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
    num_envs: int = 1,
    learning_rate: float = 1e-4,
    seed: int = 0,
    unroll_length: int = 10,
    batch_size: int = 32,
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
  
  train_data = data_sequence(type_split_every,type_size,train_data)

  data_length = len(train_data)

  xt = time.time()

  process_count = jax.process_count()
  process_id = jax.process_index()
  local_device_count = jax.local_device_count()
  local_devices_to_use = local_device_count
  
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

  assert num_envs % device_count == 0

  v_randomization_fn = None
  if randomization_fn is not None:
    randomization_batch_size = num_envs // local_device_count
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
  key_envs = jax.random.split(key_env, num_envs // process_count)
  key_envs = jp.reshape(key_envs,
                         (local_devices_to_use, -1) + key_envs.shape[1:])

  env_state = reset_fn(key_envs,None)

  net_optimizer = optax.adam(learning_rate=learning_rate)
  type_optimizer = optax.adam(learning_rate=learning_rate)

  in_net = make_in_part(
      input_size = environment.action_size + type_size,
      output_size = environment.action_size)
  
  out_net = make_out_part(
      input_size = environment.observation_size + type_size,
      output_size = environment.observation_size)
  
  loss_fn = functools.partial(
    compute_env_loss,
    network=(in_net, out_net),
    env=env,
    unroll_length=unroll_length,
    train_data=train_data
    )

  net_gradient_update_fn = gradient_update(loss_fn, net_optimizer, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True)
  type_gradient_update_fn = gradient_update(loss_fn, type_optimizer, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True)
  
  def minibatch_step(carry, xs):
    net_optimizer_state, type_optimizer_state, net_params, type_params, normalizer_params, data_chunk_id, key = carry
    key, key_loss= jax.random.split(key)

    (_, out_data), net_params, net_optimizer_state = net_gradient_update_fn(
        net_params,
        type_params,
        data_chunk_id,
        normalizer_params,
        key_loss,
        optimizer_state=net_optimizer_state)
    
    (_, out_data), type_params, type_optimizer_state = type_gradient_update_fn(
        net_params,
        type_params,
        data_chunk_id,
        normalizer_params,
        key_loss,
        optimizer_state=type_optimizer_state)

    return (net_optimizer_state, type_optimizer_state, net_params, type_params, out_data['normalizer_params'], data_chunk_id+1, key), out_data['loss_metrics']

  def training_step(
      carry: Tuple[TrainingState, int, PRNGKey],
      unused_t) -> Tuple[Tuple[TrainingState, int, PRNGKey], Metrics]:
    training_state, data_chunk_id, key = carry
    key_grad, new_key = jax.random.split(key, 2)    

    (net_optimizer_state, type_optimizzer_state, net_params, type_params, norm_params, data_chunk_id, _), loss_metrics = jax.lax.scan(
        minibatch_step,
        (training_state.net_optimizer_state, training_state.type_optimizer_state, training_state.params, training_state.type_params, training_state.normalizer_params, data_chunk_id, key_grad), (),
        length=num_minibatches)
    
    new_training_state = TrainingState(
      net_optimizer_state=net_optimizer_state,
      type_optimizer_state=type_optimizzer_state,
      params=net_params,
      type_params=type_params,
      normalizer_params=norm_params,
      env_steps=0)
    
    return (new_training_state, data_chunk_id, new_key), loss_metrics

  def training_epoch(training_state: TrainingState, data_chunk_id: int,
                     key: PRNGKey) -> Tuple[TrainingState, envs.State, Metrics]:
    (training_state, data_chunk_id, _), loss_metrics = jax.lax.scan(
        training_step, (training_state, data_chunk_id, key), (),
        length=data_length)

    loss_metrics = jax.tree_util.tree_map(jp.mean, loss_metrics)
    return training_state, data_chunk_id, loss_metrics

  training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

  # Note that this is NOT a pure jittable method.
  def training_epoch_with_timing(
      training_state: TrainingState, data_chunk_id: int,
      key: PRNGKey) -> Tuple[TrainingState, envs.State, Metrics]:

    t = time.time()
    training_state = _strip_weak_type(training_state)
    result = training_epoch(training_state, data_chunk_id, key)
    training_state, data_chunk_id, metrics = _strip_weak_type(result)

    metrics = jax.tree_util.tree_map(jp.mean, metrics)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)
    
    return training_state, data_chunk_id, metrics  # pytype: disable=bad-return-type  # py311-upgrade
  
  net_params = (
    in_net.init(key_in),
    out_net.init(key_out),
  )

  type_params = train_data.get_params(key_type)

  training_state = TrainingState(  # pytype: disable=wrong-arg-types  # jax-ndarray
      net_optimizer_state=net_optimizer.init(net_params),
      type_optimizer_state=type_optimizer.init(type_params),
      params=net_params,
      type_params=type_params,
      normalizer_params=(running_statistics.init_state(
          specs.Array(environment.action_size, jp.dtype('float32'))),
          running_statistics.init_state(
          specs.Array(environment.observation_size, jp.dtype('float32')))),
      env_steps=0)
  
  training_state = jax.device_put_replicated(
      training_state,
      jax.local_devices()[:local_devices_to_use])

  for it in range(data_loops):
    data_chunk_id = jp.zeros(1)
    logging.info('starting iteration %s %s', it, time.time() - xt)

    epoch_key, local_key = jax.random.split(local_key)
    epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
    (training_state, data_chunk_id, training_metrics) = (
        training_epoch_with_timing(training_state, data_chunk_id, epoch_keys)
    )

    key_envs = jax.vmap(
        lambda x, s: jax.random.split(x[0], s),
        in_axes=(0, None))(key_envs, key_envs.shape[1])

  # If there was no mistakes the training_state should still be identical on all
  # devices.
  pmap.assert_is_replicated(training_state)
  pmap.synchronize_hosts()
  return _unpmap(training_state.normalizer_params), _unpmap(training_state.params)