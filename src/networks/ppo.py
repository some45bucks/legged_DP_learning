from jax import numpy as jp
from typing import Any, Tuple
from brax.training import distribution
from networks.networks import Network
from networks.networks import make_feed_forward
import flax

@flax.struct.dataclass
class ppo_network:
    head_network: Network
    policy_network: Network
    value_network: Network
    action_distribution: distribution.ParametricDistribution

@flax.struct.dataclass
class ppo_network_params:
    head: jp.ndarray
    policy: jp.ndarray
    value: jp.ndarray

def make_ppo_network(head_name,input,output,head_params,value_params,policy_params) -> ppo_network:
    
    ppo_distribution = distribution.NormalTanhDistribution(event_size=output)

    if head_name == 'lstm':
        print("LSTM network head")
        raise NotImplementedError("LSTM network head not implemented")
    elif head_name == 'transformer':
        print("Transformer network head")
        raise NotImplementedError("Transformer network head not implemented")
    else:
        print("Defaulting to Feed Forward network head")

        head_network = make_feed_forward(input,name=head_name,**head_params)

    policy_network = make_feed_forward(head_network.shape[1],output_size=ppo_distribution.param_size,name='policy',**policy_params)
    value_network = make_feed_forward(head_network.shape[1],output_size=1,name='value',**value_params)

    return ppo_network(
        head_network, 
        policy_network, 
        value_network, 
        ppo_distribution)

class infrence_fn():

    deterministic = False

    def __init__(self, ppo_network: ppo_network) -> None:
        self.ppo_network = ppo_network

    def starting_hidden_state(self, batch_size: int) -> jp.ndarray:
        return None

    def __call__(self, ppo_params: ppo_network_params):

        def policy(observations: jp.ndarray, hidden: jp.ndarray, key: jp.ndarray) -> Tuple[jp.array, jp.array, Any]:

            x, new_hidden = self.ppo_network.head_network.apply(ppo_params.head, observations, hidden)
            logits, _ = self.ppo_network.policy_network.apply(ppo_params.policy, x)

            if self.deterministic:
                return self.ppo_network.action_distribution.mode(logits), new_hidden, {}
            
            raw_actions = self.ppo_network.action_distribution.sample_no_postprocessing(logits, key)
            
            log_prob = self.ppo_network.action_distribution.log_prob(logits, raw_actions)

            postprocessed_actions = self.ppo_network.action_distribution.postprocess(raw_actions)
            
            return postprocessed_actions, new_hidden, {'log_prob': log_prob,'raw_action': raw_actions}

        return policy

