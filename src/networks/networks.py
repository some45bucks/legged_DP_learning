from jax import numpy as jp
from typing import Callable, Sequence, Any, Optional
import flax
from flax import linen
import dataclasses

from networks.feed_forward import FeedForward

@dataclasses.dataclass
class Network:
  shape: Sequence[int]
  hasHiddenState: bool
  init: Callable[..., Any]
  apply: Callable[..., Any]

def activation_fn_selector(activation: str):
  if activation == 'relu':
    return linen.relu
  elif activation == 'tanh':
    return linen.tanh
  elif activation == 'sigmoid':
    return linen.sigmoid
  else:
    raise ValueError(f'Unknown activation function: {activation}')

def make_feed_forward(
    input_size: int,
    output_size: int = None,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: str = 'relu',
    activate_final: bool = False,
    name: str = 'dense'
    ) -> Network:
  
  activation_fn = activation_fn_selector(activation)

  if output_size == None:
    layers = list(hidden_layer_sizes)
    output_size = hidden_layer_sizes[-1]
  else:
    layers = list(jp.concat((jp.array(hidden_layer_sizes),jp.array([output_size]))))

  policy_module = FeedForward(name=name,layer_sizes=layers,activation=activation_fn,activate_final=activate_final)

  def apply(params, hidden, data):
    return policy_module.apply(params, data), hidden

  dummy_input = jp.zeros((1,1,input_size))

  new_feed_forward = Network(
      shape=(input_size, output_size),
      hasHiddenState=False,
      init=lambda key: policy_module.init(key, dummy_input),
      apply=apply)
  
  return new_feed_forward