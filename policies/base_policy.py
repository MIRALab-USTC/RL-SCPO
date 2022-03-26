from abc import ABC, abstractmethod
import torch
from torch.nn.functional import tanh
from torch.nn import Module
from contextlib import contextmanager

from torch_utils.base_network import BaseNetwork
from torch_utils.utils import get_numpy, from_numpy, get_nonlinearity
from utils.logger import logger
from utils.misc_utils import nullcontext

class BasePolicy(Module, ABC):

    def __init__(self, env, obs_processor=None):
        super().__init__()
        self._env = env
        observation_shape = env.observation_space.shape
        if obs_processor is None:
            self.processed_obs_shape = observation_shape
            self._obs_processor = lambda x: x
        else:
            self.processed_obs_shape = obs_processor.output_shape
            self._obs_processor = obs_processor

        self.action_shape = env.action_space.shape
        assert len(self.action_shape) == 1 and len(self.processed_obs_shape) == 1

    @abstractmethod
    def _action(self, obs, **kwargs):
        """Compute actions given processed observations"""
        raise NotImplementedError

    @contextmanager
    def with_target_network(self):
        network = self.network
        self.network = self.target_network
        yield
        self.network = network

    def action(self, obs, target=False, fast=False, **kwargs):
        processed_obs = self._obs_processor(obs)
        with self.with_target_network() if target else nullcontext():
            with torch.no_grad() if fast else nullcontext():
                action, info = self._action(processed_obs, **kwargs)
        return action, info

    def action_np(self, obs, **kwargs):
        obs = from_numpy(obs)
        with torch.no_grad():
            action, info = self.action(obs, **kwargs)
        action_np = get_numpy(action)
        info = {k:get_numpy(v) for k,v in info.items() if torch.is_tensor(v)}
        return action_np, info

    def reset(self):
        pass

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

class BaseStochasticPolicy(BasePolicy):

    def __init__(self, env, obs_processor=None):
        super().__init__(env, obs_processor)

    @torch.no_grad()
    def action_eval(self, obs, **kwargs):
        processed_obs = self._obs_processor(obs)
        action, info = self._action_eval(processed_obs, **kwargs)
        return action, info

    def action_eval_np(self, obs, **kwargs):
        obs = from_numpy(obs)
        action, info = self.action_eval(obs, **kwargs)
        action_np = get_numpy(action)
        info_np = {k:get_numpy(v) for k,v in info.items() if torch.is_tensor(v)}
        return action_np, info_np

    @abstractmethod
    def _action_eval(self, obs):
        raise NotImplementedError


BasePolicy.register(BaseStochasticPolicy)
