import jax
from jax import numpy as jp
import numpy as np
from typing import Tuple, Union

import functools
import time
from typing import Callable, Optional, Tuple, Union

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

from networks.ppo import ppo_network, ppo_network_params, infrence_fn
from train.evaluator import evaluator
from train.gradients import gradient_update_fn as gradient_update
from train.losses import compute_ppo_loss
from envs.custom_wrappers import HiddenStateWrapper

@flax.struct.dataclass
class TrainingState:
  optimizer_state: optax.OptState
  params: ppo_network_params
  normalizer_params: running_statistics.RunningStatisticsState
  env_steps: jp.ndarray