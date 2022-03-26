from abc import ABC, abstractmethod
import torch
from torch.nn import Module
from contextlib import contextmanager

from torch_utils.utils import get_numpy
from utils.logger import logger
from utils.misc_utils import nullcontext
from torch_utils.utils import soft_update_from_to


class BaseValue(Module, ABC):

    def __init__(self, env, obs_processor=None):
        super().__init__()
        self._env = env
        self.action_shape = env.action_space.shape
        observation_shape = env.observation_space.shape

        if obs_processor is None:
            self.processed_obs_shape = observation_shape
            self._obs_processor = lambda x: x
        else:
            self.processed_obs_shape = obs_processor.output_shape
            self._obs_processor = obs_processor
        assert len(self.action_shape) == 1 and len(self.processed_obs_shape) == 1

    @abstractmethod
    def _value(self, inputs, **kwargs):
        raise NotImplementedError

    def get_snapshot(self):
        return self.state_dict()

    def load_snapshot(self, state_dict):
        self.load_state_dict(state_dict)

    def get_diagnostics(self):
        return {}

    def has_nan(self):
        for net in self.children():
            if isinstance(net, BaseNetwork) and net.has_nan():
                return True
            else:
                raise NotImplementedError
        return False

    @contextmanager
    def with_target_network(self):
        network = self.network
        self.network = self.target_network
        yield
        self.network = network

    def update_target(self, tau):
        if self.with_target_value:
            soft_update_from_to(self.network, self.target_network, tau)
        else:
            warnings.warn("target network do not exists")

class BaseQValue(BaseValue):

    def value(self, obs, action, target=False, fast=False, **kwargs):
        processed_obs = self._obs_processor(obs)
        inputs = torch.cat((processed_obs, action), dim=-1)
        with self.with_target_network() if target else nullcontext():
            with torch.no_grad() if fast else nullcontext():
                value, info = self._value(inputs, **kwargs)
        return value, info

class BaseStateValue(BaseValue):

    def value(self, obs, target=False, fast=False, **kwargs):
        inputs = self._obs_processor(obs)
        with self.with_target_network() if target else nullcontext():
            with torch.no_grad() if fast else nullcontext():
                value, info = self._value(inputs, **kwargs)
        return value, info
