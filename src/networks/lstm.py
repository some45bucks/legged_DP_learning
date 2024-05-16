from jax import numpy as jp
from typing import Callable, Sequence, Any
from flax import linen
from typing import Optional

class LSTM(linen.Module):
    
    size: int
    kernel_init: Callable[...,Any] = linen.initializers.lecun_normal()
    bias_init: Callable[...,Any] = linen.initializers.zeros
    name: str = 'lstm'
    learnable_hidden_state: bool = True

    def setup(self):
        if self.learnable_hidden_state:
          self.initial_h = self.param('initial_h', linen.initializers.zeros, (1, self.size))
          self.initial_c = self.param('initial_c', linen.initializers.zeros, (1, self.size))
        else:
          self.initial_h = jp.zeros((1, self.size))
          self.initial_c = jp.zeros((1, self.size))

    def initialize_carry(self, rng: Optional[jp.ndarray] = None, shape: Optional[Sequence[int]] = None):
        return (self.initial_h, self.initial_c)
    
    @linen.compact
    def __call__(self, data, carry):
        if carry is None:
            carry = self.initialize_carry()
        carry, z = linen.OptimizedLSTMCell(
          self.size,
          kernel_init=self.kernel_init,
          bias_init=self.bias_init,
          name=f'{self.name}')(carry, data)
        return z, carry