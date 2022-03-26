import numpy as np
from numpy.random import uniform
from math import log

from policies.base_policy import BasePolicy
from torch_utils.utils import device


class UniformlyRandomPolicy(BasePolicy):
    def __init__(self, env):
        super().__init__(env)
        self.low = self._env.action_space.low
        self.high = self._env.action_space.high
        self.shape = self.low.shape[0]
        self.log_prob = - self.shape * log(2)

    def _action(self):
        pass

    def action(self):
        raise NotImplementedError

    def action_np(self, obs, return_log_prob=False, **kwargs):
        info = {}
        action = uniform(self.low, self.high, (obs.shape[0], self.shape))
        if return_log_prob:
            info["log_probs"] = np.full((obs.shape[0], 1), self.log_prob, dtype=np.float32)
        return action, info

    def to(self, my_device):
        pass
