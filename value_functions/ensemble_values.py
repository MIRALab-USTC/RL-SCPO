import torch
from contextlib import contextmanager
import warnings
from contextlib import contextmanager

from torch.nn import Module

from value_functions.base_value import BaseQValue
from torch_utils.ensemble_network import EnsembleMlpNetwork
from torch_utils.utils import soft_update_from_to
from utils.logger import logger
from utils.misc_utils import nullcontext


class EnsembleQValue(BaseQValue):
    def __init__(
                self,
                env,
                ensemble_size=2,
                obs_processor=None,
                with_target_value=True,
                value_name="ensemble_q_value",
                **mlp_kwargs):
        super().__init__(env, obs_processor)
        self.with_target_value = with_target_value
        self.ensemble_size = ensemble_size
        self.value_name = value_name
        # assert len(self.processed_obs_shape) == 1 and len(self.action_shape) == 1

        self.network = EnsembleMlpNetwork(self.processed_obs_shape[0] + self.action_shape[0],
                                          1,
                                          ensemble_size=self.ensemble_size,
                                          network_name=self.value_name,
                                          **mlp_kwargs)

        if self.with_target_value:
            self.target_network = EnsembleMlpNetwork(self.processed_obs_shape[0] + self.action_shape[0],
                                                     1,
                                                     ensemble_size=self.ensemble_size,
                                                     network_name=f"{self.value_name}_target",
                                                     **mlp_kwargs)
            soft_update_from_to(self.network, self.target_network, 1.)

    def _value(self, inputs, return_min=False, return_max=False):
        info_dict = {}
        value_cat = self.network(inputs)
        if return_min:
            info_dict["value_min"] = torch.min(value_cat, dim=-1, keepdim=True)[0]
        if return_max:
            info_dict["value_max"] = torch.max(value_cat, dim=-1, keepdim=True)[0]
        return value_cat, info_dict

    def _value_single(self, inputs, i_th_net=0, return_min=True): # return min not really used when single value
        value = self.network.single_forward(inputs, i_th_net)
        return None, {"value_min": value}

    def value(self, obs, action, single_value=False, **kwargs):
        with self.with_single_value() if single_value else nullcontext():
            return super().value(obs, action, **kwargs)

    @contextmanager
    def with_single_value(self):
        value_function = self._value
        self._value = self._value_single
        yield
        self._value = value_function