
import torch
from torch_utils.utils import from_numpy, get_numpy

class MakeDeterministic():
    def __init__(self, policy):
        self.policy = policy

    def action(self, obs):
        raise NotImplementedError

    def action_np(self, obs, **kwargs):
        return self.policy.action_eval_np(obs, **kwargs)

    def reset(self):
        pass
