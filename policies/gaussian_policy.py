import torch
from torch.nn.functional import tanh
from math import pi
from torch.nn import Module
import numpy as np

from torch_utils.policy_networks import MeanLogstdNet
import torch_utils.utils as ptu
from torch_utils.utils import normal_distribution, get_nonlinearity
from policies.base_policy import BaseStochasticPolicy
from utils.policy_utils import LOG_STD_MIN, LOG_STD_MAX, SAFE_ACTION_MIN, SAFE_ACTION_MAX


class GaussianPolicy(BaseStochasticPolicy):
    def __init__(self, env, obs_processor=None, output_processor="tanh", **mlp_kwargs):
        super().__init__(env, obs_processor)
        self.network = MeanLogstdNet(self.processed_obs_shape[0],
                                     self.action_shape[0],
                                     **mlp_kwargs)
        self.gassian_distribution = normal_distribution(np.zeros(self.action_shape[0]), np.ones(self.action_shape[0]))
        self.output_processor = get_nonlinearity(output_processor)

        self._set_constant()

    def _set_constant(self):
        self.constant1 = (- 0.5 * torch.log(2 * torch.tensor(pi))).to(ptu.device)

    def get_mean_log_std(self, obs):
        raw_action, log_std = self.network(obs)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return raw_action, log_std

    def _action(self, obs, return_log_prob=False):
        info = {}
        raw_action, log_std = self.get_mean_log_std(obs)
        std = torch.exp(log_std)
        noise = self.gassian_distribution.sample((raw_action.shape[0],))
        action = raw_action + noise * std
        action_processed = self.output_processor(action)
        if return_log_prob:
            info["log_probs"] = self.get_log_prob(action_processed, log_std, noise)
        return action_processed, info

    def get_log_prob(self, action_processed, log_std, noise):
        """the sample entropy equals to the log_prob"""
        log_p = self.constant1 - log_std - 0.5 * noise * noise - torch.log(1 - action_processed * action_processed + 1e-6)
        log_p = torch.sum(log_p, dim=-1, keepdim=True)
        return log_p

    def _action_eval(self, obs):
        action, _ = self.network(obs)
        action_processed = self.output_processor(action)
        return action_processed, {}

    def get_log_prob_from_state_action(self, obs, actions):
        # orig_actions = 0.5 * (torch.log( 1 + actions + 1e-6 ) - torch.log(1 - actions + 1e-6))
        safe_actions = torch.clamp(actions, min=SAFE_ACTION_MIN, max=SAFE_ACTION_MAX)
        orig_actions = torch.atanh(safe_actions)
        with torch.no_grad():
            mean, log_std = self.network(obs)
        std = torch.exp(log_std)

        log_p = self.constant1 - log_std - 0.5 * ((orig_actions - mean) / (std + 1e-6)) ** 2 - torch.log(1 - actions ** 2 +1e-6)
        log_p = torch.sum(log_p, dim=-1, keepdim=True)
        return log_p
