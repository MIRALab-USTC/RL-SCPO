import torch
from torch import nn

from torch_utils.utils import get_nonlinearity, normal_distribution
from torch_utils.base_network import BaseNetwork
from torch_utils.mlp_network import MlpNetwork


class EnsembleMlpNetwork(BaseNetwork):
    def __init__(
                self,
                input_size,
                output_size,
                ensemble_size=2,
                hidden_layer=[128, 128],
                activation="relu",
                output_activation=None,
                network_name="ensemble_mlp"):
        super().__init__(network_name)
        self.input_size = input_size
        self.output_size = output_size
        self.ensemble_size = ensemble_size
        self.hidden_layer = hidden_layer
        self.activation = activation
        self.output_activation = output_activation

        self.network_list = []
        for i in range(self.ensemble_size):
            mlp_network = MlpNetwork(self.input_size, 
                                     self.output_size, 
                                     hidden_layer=self.hidden_layer, 
                                     activation=self.activation, 
                                     output_activation=self.output_activation,
                                     network_name=f"ensemble_{i}")
            setattr(self, f"network_{i}", mlp_network)
            self.network_list.append(mlp_network)

    def forward(self, inputs):
        result_list = [network(inputs) for network in self.network_list]
        result = torch.cat(result_list, dim=-1)
        return result

    def single_forward(self, inputs, i_th_net=0):
        return self.network_list[i_th_net](inputs)
