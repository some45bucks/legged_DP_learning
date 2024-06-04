from typing import Sequence
import jax
from jax import numpy as jp
from flax import linen
from utils.save_load import load_rollout

def gather_rollout_data(folder_path, num_rollouts):

    piplines = []
    actions = []
    for i in range(num_rollouts):
        pipeline, action = load_rollout(folder_path+f'rollout_{i}.pkl')
        piplines += pipeline
        actions += action
     
    return piplines, actions

def extract_q_dq(data):

        out_data = []

        for i in range(len(data)):
            q, dq = data[i].q, data[i].qd
            out_data.append((q,dq))

        return out_data

class data_sequence(Sequence):
    def __init__(self, chunk_size, unroll_length, type_size, data):
        assert len(data[0]) == len(data[1]), "Data and actions must be the same length" 
        # assert (len(data[0])-unroll_length) % chunk_size == 0, f"Data length ({len(data[0])}) minus unroll length ({unroll_length}) ({len(data[0])-unroll_length})  must be divisible by chunk size ({chunk_size}) instead has remainder {(len(data[0])-unroll_length) % chunk_size}"
        if len(data[0]) % chunk_size != 0:
            for i in range(chunk_size - (len(data[0]) % chunk_size)):
                data[0].append(data[0][-1])
                data[1].append(data[1][-1])
        self.unroll_length = unroll_length
        self.chunk_size = chunk_size
        self.chunked_data = [(extract_q_dq(data[0][i:i+chunk_size+unroll_length]),data[1][i:i+chunk_size+unroll_length],extract_q_dq([data[0][i]])[0]) for i in range(0, len(data[0])-unroll_length, chunk_size)]
        self.type_shape = (len(self.chunked_data), type_size)
        self.types = type_params(size=self.type_shape[1], num=self.type_shape[0])

    def flat_stack(self, data):
        flat_data = []
        for i in range(len(data)):
            flat,unflat = jax.tree_util.tree_flatten(data[i])
            flat_data.append(flat)
        
        _flat_data = []
        for i in range(len(flat_data[-1])):
            
           
            try:
                _stack = [flat_data[j][i] for j in range(len(flat_data))]
                _stack = jp.stack(_stack)
            except:
                _stack = [flat_data[-1][i] for j in range(len(flat_data))]
                _stack = jp.stack(_stack)

            # _stack = jp.stack(_stack)
            if len(_stack.shape) == 1:
                _stack = _stack.reshape(-1,1)
            _flat_data.append(_stack)

        return unflat.unflatten(_flat_data)

    def get_type(self, params, i):
        return self.types.apply(params, i)

    def get_params(self, key):
        return self.types.init(key, 0)
    
    def __getitem__(self, index):
        chunk = self.flat_stack([self.flat_stack(self.chunked_data[index])])
        return self.flat_stack([(self.flat_stack(chunk[0]), self.flat_stack(chunk[1]), chunk[2])])

    def __len__(self):
        return len(self.chunked_data)
    
class type_params(linen.Module):

    size: int
    num: int

    def setup(self):
        self.types = self.param(f'0', linen.initializers.uniform(), (self.num, self.size))

    @linen.compact
    def __call__(self, i):
        return self.types[i]