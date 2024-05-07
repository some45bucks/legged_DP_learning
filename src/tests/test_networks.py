import unittest
import jax
from src.networks.networks import make_feed_forward

class TestNetworks(unittest.TestCase):
    def test_create_feed_froward(self):
        mkn = make_feed_forward(3,3,hidden_layer_sizes=(4,4),activation='relu',activate_final=False)

        self.assertEqual(mkn.shape,(3,3))
        self.assertEqual(mkn.hasHiddenState,False)

        key = jax.random.PRNGKey(0)

        params = mkn.init(key, jax.numpy.ones((1,3)))

        self.assertEqual(params['params']['w1'].shape,(3,4))


