from abc import ABC, abstractmethod
from os import path as osp
import torch

from torch.nn import Module


class BaseNetwork(Module, ABC):
    @abstractmethod
    def __init__(self, network_name):
        super().__init__()
        self.network_name = network_name

    @abstractmethod
    def forward(self, inputs):
        raise NotImplementedError

    def has_nan(self):
        for parameter in self.parameters():
            if torch.any(torch.isnan(parameter)):
                return True
        return False
