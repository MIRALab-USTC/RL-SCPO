
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Module

from torch_utils.mlp_network import MlpNetwork
from torch_utils.base_network import BaseNetwork

class MeanLogstdNet(MlpNetwork):
    def __init__(
                self,
                input_size,
                output_size,
                hidden_layer=[128, 128],
                activation="relu",
                output_activation=None,
                network_name="mean_logstd"):
        super().__init__(input_size, output_size * 2, hidden_layer, activation, output_activation, network_name)

    def forward(self, inputs):
        outputs = super().forward(inputs)
        mean, log_std = torch.chunk(outputs, 2, dim=-1)
        return mean, log_std
