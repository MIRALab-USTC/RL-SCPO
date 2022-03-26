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

            discount=0.99,

            log_alpha_lr=3e-4,
            policy_lr=3e-4,
            qf_lr=3e-4,
            optimizer_class='Adam',

            soft_target_tau=5e-3,
            target_update_period=1,
            plotter=None,

            alpha_if_not_automatic=1e-2,
            use_automatic_entropy_tuning=True,
            target_entropy=None,

            state_noise=0.005
    ):
        super().__init__(env=env, policy=policy, qf=qf,
                        discount=discount,
                        log_alpha_lr=log_alpha_lr,
                        policy_lr=policy_lr,
                        qf_lr=qf_lr,
                        optimizer_class=optimizer_class,
                        soft_target_tau=soft_target_tau,
                        target_update_period=target_update_period,
                        plotter=plotter,
                        alpha_if_not_automatic=alpha_if_not_automatic,
                        use_automatic_entropy_tuning=use_automatic_entropy_tuning,
                        target_entropy=target_entropy)
        self.state_noise = tensors(state_noise)
        self.noise_weights_func = self._get_uniform_weights

    def _get_uniform_weights(self, size):
        return ones(*size)

    # def _get_q_func(self, target):
    #     return self.qf.value_target if target else self.qf.value

    def _get_algo_q_with_regularizer(self, obs, actions, log_prob_action, alpha, target=False):
        ## calculate soft q
        q_min, soft_q = super()._get_algo_q_with_regularizer(obs, actions, log_prob_action, alpha, target, fast=False)
        ## calculate gradient penalty
        gradients = grad(outputs=q_min, inputs=obs,
                         grad_outputs=self.noise_weights_func(q_min.size()), only_inputs=True, create_graph=not target)[0]
        gradient_penalty = norm(gradients, p=1, dim=-1, keepdim=True) # 换成 1 范数

        q_with_regularizer = soft_q - self.state_noise * gradient_penalty

        if not target:
            with torch.no_grad():
                self.temp_info_dict["gradient_penalty_loss"] = - self.state_noise * (gradient_penalty.mean())
        return q_min, q_with_regularizer

    def _set_obs(self, obs, next_obs):
        return obs.requires_grad_(), next_obs.requires_grad_()

    def _get_diagnostics_dict(self):
        # locals().update(self.temp_info_dict)
        diagnostics = super()._get_diagnostics_dict()
        diagnostics["gradient_penalty_loss"] = self.temp_info_dict["gradient_penalty_loss"]
        diagnostics["percentage"] = diagnostics["gradient_penalty_loss"] / diagnostics['Policy Loss']
        return diagnostics
