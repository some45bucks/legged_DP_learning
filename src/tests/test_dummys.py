import jax
from jax import numpy as jp
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath
from brax.training.types import PRNGKey
from brax.training import types
import mujoco
import unittest

class DummyEnv(PipelineEnv):
    def __init__(self,**kwargs):

        path = epath.Path('../../data/go1/go1_scene.xml').as_posix() 
        sys = mjcf.load(path)

        self._nv = sys.nv
        self._init_q = jp.array(sys.mj_model.keyframe('home').qpos)

        super().__init__(sys, backend='mjx')

        self._torso_idx = mujoco.mj_name2id(
            sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, 'trunk'
        )
    
    def reset(self, rng: PRNGKey) -> State:
        pipeline_state = self.pipeline_init(self._init_q, jp.zeros(self._nv))
        return State(pipeline_state=pipeline_state,obs=jp.zeros(1),reward=jp.zeros(1),done=jp.zeros(1))

    def step(self, state: State, action: types.Action) -> State:
        pipeline_state = self.pipeline_step(state.pipeline_state,action=action)
        done = pipeline_state.x.pos[self._torso_idx - 1, 2] < 0.18
        return State(pipeline_state=pipeline_state,obs=jp.zeros(1),reward=jp.zeros(1),done=done)

class TestDummies(unittest.TestCase):
    def test_dummy_env(self):

        env = DummyEnv()
        jit_reset = jax.jit(env.reset)
        jit_step = jax.jit(env.step)

        rng = jax.random.PRNGKey(0)
        state = jit_reset(rng)

        for i in range(1000):
            state = jit_step(state, jp.zeros(12))

            if state.done:
                break

        self.assertTrue(state.done and i > 0)
