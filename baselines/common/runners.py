import numpy as np
from abc import ABC, abstractmethod

class AbstractEnvRunner(ABC):
    def __init__(self, *, env, model, nsteps):
        self.env = env
        self.model = model
        nenv = env.num_envs
        self.batch_ob_shape = (nenv*nsteps,) + env.observation_space.shape
        self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)
        self.obs[:] = env.reset()
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]
        self.gelabel=np.zeros(nenv)
        self.gelabel1=np.zeros(nenv)
        self.sum_reward=np.zeros(nenv)
        self.label=np.zeros(nenv)
        self.pilabel=np.zeros(nenv)
        self.infosbegin=[{'ale.lives':6} for _ in range(nenv)]
        self.sum_turn_reward=0
        self.pi=np.zeros(nenv)
    @abstractmethod
    def run(self):
        raise NotImplementedError



