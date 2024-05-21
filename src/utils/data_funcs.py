from typing import Sequence
import jax
from jax import numpy as jp
from flax import linen
from utils.save_load import load_rollout

def gather_rollout_data(folder_path, num_rollouts):
    data = []
    for i in range(num_rollouts):
        rollout = load_rollout(folder_path+f'rollout_{i}.pkl')
        data += rollout
    
    return data

class data_sequence(Sequence):
    def __init__(self, chunk_size, type_size, data):
        self.chunked_data = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
        self.types = type_params(size=type_size, num=len(self.chunked_data))

    def get_type(self, params, i):
        return self.types.apply(params, i)

    def get_params(self, key):
        return self.types.init(key, 0)

    def __getitem__(self, index):
        return self.chunked_data[index]

    def __len__(self):
        return len(self.chunked_data)
    
class type_params(linen.Module):

    size: int
    num: int

    def setup(self):
        self.types = [self.param(f'type_{i}', linen.initializers.uniform(), (1, self.size)) for i in range(self.num)]

    @linen.compact
    def __call__(self, i):
        return self.types[i]