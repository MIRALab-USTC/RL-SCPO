import torch
from torch.nn.functional import tanh
from math import pi
from torch.nn import Module
import numpy as np

from torch_utils.mlp_network import MlpNetwork
import torch_utils.utils as ptu
from torch_utils.utils import normal_distribution, get_nonlinearity, soft_update_from_to, from_numpy
from policies.base_policy import BaseStochasticPolicy
from utils.policy_utils import ACTION_MIN, ACTION_MAX

class DeterministicPolicyWithNoise(BaseStochasticPolicy):
    def __init__(self, env, obs_processor=None, output_processor="tanh", expl_noise=0.1, use_target=True, **mlp_kwargs):
        super().__init__(env, obs_processor)
        self.network = MlpNetwork(self.processed_obs_shape[0],
                                  self.action_shape[0],
                                  output_activation=output_processor,
                                  network_name="deterministic_policy_with_noise",
                                  **mlp_kwargs)
        self.use_target = use_target
        if use_target:
            self.target_network = MlpNetwork(self.processed_obs_shape[0],
                                            self.action_shape[0],
                                            output_activation=output_processor,
                                            network_name="deterministic_policy_with_noise_target",
                                            **mlp_kwargs)
            soft_update_from_to(self.network, self.target_network, 1.)
        self.gassian_distribution = normal_distribution(np.zeros(self.action_shape[0]), np.ones(self.action_shape[0]))

        self.expl_noise = expl_noise


    def _action(self, obs, noise=None, noise_clamp=0.5):
        action = self.network(obs)
        if noise is None:
            action_noise = ACTION_MAX * self.expl_noise * self.gassian_distribution.sample((len(action),))
            action = (action + action_noise).clamp(ACTION_MIN, ACTION_MAX)
        elif noise > 0:
            action_noise = (noise * self.gassian_distribution.sample((len(action),))).clamp(-noise_clamp, noise_clamp)
            action = (action + action_noise).clamp(ACTION_MIN, ACTION_MAX)
        return action, {}

    def _action_eval(self, obs):
        return self.network(obs), {}

    def update_target(self, tau):
        if self.use_target:
            soft_update_from_to(self.network, self.target_network, tau)
        else:
            warnings.warn("target network do not exists")
