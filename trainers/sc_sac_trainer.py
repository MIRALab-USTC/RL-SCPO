import torch
from torch import optim, norm
from torch.autograd import grad
import numpy as np
from collections import OrderedDict

from trainers.sac_trainer import SACTrainer
import torch_utils.utils as ptu
from torch_utils.utils import ones, tensors
from utils.eval_utils import create_stats_ordered_dict

class SCSACTrainer(SACTrainer):

    def __init__(
            self,
            env,
            policy,
            qf,

            num_trains_per_batch=1,

            discount=0.99,

            log_alpha_lr=3e-4,
            policy_lr=3e-4,
            qf_lr=3e-4,
            optimizer_class='Adam',

            soft_target_tau=5e-3,
            target_update_period=1,

            alpha_if_not_automatic=1e-2,
            use_automatic_entropy_tuning=True,
            target_entropy=None,

            state_noise=0.005
    ):
        super().__init__(env=env, policy=policy, qf=qf,
                        num_trains_per_batch=num_trains_per_batch,
                        discount=discount,
                        log_alpha_lr=log_alpha_lr,
                        policy_lr=policy_lr,
                        qf_lr=qf_lr,
                        optimizer_class=optimizer_class,
                        soft_target_tau=soft_target_tau,
                        target_update_period=target_update_period,
                        alpha_if_not_automatic=alpha_if_not_automatic,
                        use_automatic_entropy_tuning=use_automatic_entropy_tuning,
                        target_entropy=target_entropy)
        self._state_noise = state_noise
        self.noise_weights_func = self._get_uniform_weights

    def _set_obs(self, obs, next_obs):
        return obs.requires_grad_(), next_obs.requires_grad_()

    def _get_uniform_weights(self, size):
        return ones(*size)

    def _get_algo_q_with_regularizer(self, obs, actions, log_prob_action, alpha, target=False):
        ## calculate soft q
        q_min, soft_q = super()._get_algo_q_with_regularizer(obs, actions, log_prob_action, alpha, target, fast=False)
        ## calculate gradient penalty
        gradients = grad(outputs=q_min, inputs=obs,
                         grad_outputs=self.noise_weights_func(q_min.size()), only_inputs=True, create_graph=not target)[0]
        gradient_norm = norm(gradients, p=1, dim=-1, keepdim=True) # 换成 1 范数

        state_noise = self._state_noise

        q_with_regularizer = soft_q - state_noise * gradient_norm

        if not target:
            self.training_records["state_noise"] = state_noise
            self.temp_info["gradient_norm"] = gradient_norm
        return q_min, q_with_regularizer

    @torch.no_grad()
    def _update_diagnostics(self):
        super()._update_diagnostics()
        self.diagnostics["gradient_norm"] = gradient_norm = self.temp_info["gradient_norm"].mean().item()
        self.diagnostics["gradient_penalty"] = gradient_penalty = - self.diagnostics["state_noise"] * gradient_norm
        self.diagnostics["gradient_percentage"] = gradient_penalty / self.diagnostics["policy_loss_without_entropy"]
