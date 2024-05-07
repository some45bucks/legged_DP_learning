from jax import numpy as jp
from typing import Callable, Sequence, Any
from flax import linen

class FeedForward(linen.Module):
    layer_sizes: Sequence[int]
    activation: Callable[[jp.ndarray], jp.ndarray] = linen.relu
    activate_final: bool = False
    kernel_init: Callable[...,Any] = linen.initializers.lecun_normal()
    bias_init: Callable[...,Any] = linen.initializers.zeros
    name: str = 'dense'

    @linen.compact
    def __call__(self, data):
        z = data
        for i,size in enumerate(self.layer_sizes):
            z = linen.Dense(
              size,
              kernel_init=self.kernel_init,
              bias_init=self.bias_init,
              name=f'{self.name}_{i}')(z)
            if i != len(self.layer_sizes) - 1 or self.activate_final:
              z = self.activation(z)
        return z