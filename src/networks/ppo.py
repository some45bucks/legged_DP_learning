from jax import numpy as jp
from typing import Any, Tuple, Sequence, Callable
from brax.training import distribution
from networks.networks import Network
from networks.networks import make_feed_forward, make_lstm
from brax.training.acme import running_statistics
import flax

@flax.struct.dataclass
class ppo_network:
    has_hidden_state: bool
    normalizer: Callable[..., Any]
    head_network: Network
    policy_network: Network
    value_network: Network
    action_distribution: distribution.ParametricDistribution

@flax.struct.dataclass
class ppo_network_params:
    head: jp.ndarray
    policy: jp.ndarray
    value: jp.ndarray

def make_ppo_network(head_name,input,output,head_params,value_params,policy_params,normalizer) -> ppo_network:
    
    ppo_distribution = distribution.NormalTanhDistribution(event_size=output)

    if head_name == 'lstm':
        print("LSTM network head")
        head_network = make_lstm(input,name=head_name,**head_params)
    elif head_name == 'transformer':
        print("Transformer network head")
        raise NotImplementedError("Transformer network head not implemented")
    else:
        print("Defaulting to Feed Forward network head")
        head_network = make_feed_forward(input,name=head_name,**head_params)

    policy_network = make_feed_forward(head_network.shape[1],output_size=ppo_distribution.param_size,name='policy',**policy_params)
    value_network = make_feed_forward(head_network.shape[1],output_size=1,name='value',**value_params)

    return ppo_network(
        head_network.hasHiddenState,
        normalizer,
        head_network, 
        policy_network, 
        value_network, 
        ppo_distribution)


def make_ppo_policy(ppo_network: ppo_network,normalizer_params: running_statistics.RunningStatisticsState ,ppo_params: ppo_network_params, key: jp.ndarray):

    [key_hold] = key

    def policy(observations: jp.ndarray, hidden: jp.ndarray) -> Tuple[jp.array, jp.array, Any]:

        observations = ppo_network.normalizer(observations, normalizer_params)
        x, new_hidden = ppo_network.head_network.apply(ppo_params.head, hidden, observations)
        logits, _ = ppo_network.policy_network.apply(ppo_params.policy, None, x)

        key_hold[0], use = jp.split(key_hold[0], 2)
        
        raw_actions = ppo_network.action_distribution.sample_no_postprocessing(logits, use)

        postprocessed_actions = ppo_network.action_distribution.postprocess(raw_actions)
        
        return postprocessed_actions, new_hidden

    return policy

