import torch
from torch import optim
import numpy as np
from collections import OrderedDict

from trainers.base_trainer import BaseBatchTrainer
import torch_utils.utils as ptu
from torch_utils.utils import ones, zeros, tensors
from utils.eval_utils import create_stats_ordered_dict
from utils.misc_utils import nullcontext


class SACTrainer(BaseBatchTrainer):

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
    ):
        super().__init__(num_trains_per_batch)
        self.env = env
        self.policy = policy
        self.qf = qf
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.discount = tensors(discount)

        self.optimizer_class = getattr(optim, optimizer_class)
        self.qf_lr = qf_lr
        self.policy_lr = policy_lr
        self.log_alpha_lr = log_alpha_lr

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy is None:
                target_entropy = -np.prod(self.env.action_space.shape).item()  # heuristic value from Tuomas
            self.target_entropy = tensors(target_entropy)
            self.log_alpha = zeros(1, requires_grad=True)
            self.alpha_optimizer = self.optimizer_class(
                [self.log_alpha],
                lr=log_alpha_lr,
            )
        else:
            self.alpha_if_not_automatic = tensors(alpha_if_not_automatic)
        self.policy_optimizer = self.optimizer_class(
            self.policy.network.parameters(),
            lr=policy_lr,
        )
        self.qf_optimizer = self.optimizer_class(
            self.qf.network.parameters(),
            lr=qf_lr,
        )

    def _update_target_networks(self):
        if self._num_train_steps % self.target_update_period == 0:
            self.qf.update_target(self.soft_target_tau)

    def _train_temperature_alpha(self, obs, actions, rewards, next_obs, terminals, **kwargs):

        new_actions, actions_info = self.policy.action(obs, return_log_prob=True)
        log_prob_new_actions = actions_info["log_probs"]
        with torch.no_grad():
            average_entropy = - (log_prob_new_actions.mean())

        if self.use_automatic_entropy_tuning:
            alpha_loss = self.log_alpha * (average_entropy - self.target_entropy)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.detach().exp() # alpha is detached
        else:
            alpha_loss = 0
            alpha = self.alpha_if_not_automatic
        return alpha, alpha_loss.detach(), new_actions, log_prob_new_actions, average_entropy

    def _get_algo_q_with_regularizer(self, obs, actions, log_prob_action, alpha, target=False, fast=None):
        """this function returns soft q for sac_algorithm"""
        # q_func = self._get_q_func(target=target)
        if fast is None:
            fast = target
        _, q_info = self.qf.value(obs, actions, target=target, fast=fast, return_min=True)
        q_min = q_info["value_min"]

        q_with_regularizer = q_min - alpha * log_prob_action
        return q_min, q_with_regularizer

    def _train_critic(self, obs, actions, rewards, next_obs, terminals, alpha, **kwargs):
        # Make sure policy accounts for squashing functions like tanh correctly!

        next_action, next_policy_info = self.policy.action(next_obs, return_log_prob=True)
        log_prob_next_action = next_policy_info['log_probs']

        # get qf network outputs
        q_value_ensemble, _ = self.qf.value(obs, actions)

        # get target soft-q
        _, q_with_regularizer = self._get_algo_q_with_regularizer(obs=next_obs, actions=next_action, log_prob_action=log_prob_next_action.detach(), alpha=alpha.detach(), target=True)

        # get critic loss
        q_target = rewards + (1. - terminals) * self.discount * q_with_regularizer.detach()
        qf_loss = ((q_value_ensemble - q_target.detach()) ** 2).mean()

        # train critic
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()
        return qf_loss, q_value_ensemble, q_target

    def _train_actor(self, obs, actions, rewards, next_obs, terminals, new_actions, log_prob_new_actions, alpha, **kwargs):
        # get actor loss
        q_new_actions, q_with_regularizer = self._get_algo_q_with_regularizer(obs=obs, actions=new_actions, log_prob_action=log_prob_new_actions, alpha=alpha)
        policy_loss = (-q_with_regularizer).mean()

        # train actor
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        return policy_loss, q_new_actions

    def _set_obs(self, obs, next_obs):
        return obs, next_obs

    def _train_from_torch_batch(self, batch):
        rewards, terminals, obs, actions, next_obs = batch['rewards'], batch['terminals'], batch['observations'], batch['actions'], batch['next_observations']
        obs, next_obs = self._set_obs(obs, next_obs)
        train_dataset = {"obs": obs, "actions": actions, "rewards": rewards, "next_obs": next_obs, "terminals": terminals}

        """
        Alpha
        """

        alpha, alpha_loss, new_actions, log_prob_new_actions, average_entropy = self._train_temperature_alpha(**train_dataset)
        self.training_records["alpha"], self.training_records["alpha_loss"], self.training_records["entropy"] = alpha.item(), alpha_loss.item(), average_entropy.item()

        """
        QF 
        """

        qf_loss, self.temp_info["q_value_ensemble"], self.temp_info["q_target"] = self._train_critic(**train_dataset,  alpha=alpha)
        self.training_records["qf_loss"] = qf_loss.item()

        """
        Soft Updates
        """
        self._update_target_networks()

        """
        policy
        """
        policy_loss, self.temp_info["q_new_action"] = self._train_actor(**train_dataset, new_actions=new_actions, log_prob_new_actions=log_prob_new_actions, alpha=alpha)
        self.training_records["policy_loss"] = policy_loss.item()

    @torch.no_grad()
    def _update_diagnostics(self):
        """
        Eval should set this to None.
        This way, these statistics are only computed for one batch.
        """

        super()._update_diagnostics()

        q_value_ensemble, q_target, q_new_action = self.temp_info["q_value_ensemble"], self.temp_info["q_target"], self.temp_info["q_new_action"]

        self.diagnostics["policy_loss_without_entropy"] = policy_loss_without_entropy = q_new_action.mean().item()
        self.diagnostics["entropy_penalty"] = entropy_penalty = self.diagnostics["alpha"] * self.diagnostics["entropy"]
        self.diagnostics["entropy_percentage"] = entropy_penalty / policy_loss_without_entropy

        self.diagnostics.update(create_stats_ordered_dict(
            'Q1Pred',
            ptu.get_numpy(q_value_ensemble[:, 0]),
        ))
        self.diagnostics.update(create_stats_ordered_dict(
            'Q2Pred',
            ptu.get_numpy(q_value_ensemble[:, 1]),
        ))
        self.diagnostics.update(create_stats_ordered_dict(
            'QTargetWithReg', # that is, q target with regularizer
            ptu.get_numpy(q_target),
        ))
        self.diagnostics.update(create_stats_ordered_dict(
            'PolicyLossWithoutReg', # that is, q new action without regularizer
            ptu.get_numpy(q_new_action),
        ))

    @property
    def networks(self):
        return [
            self.policy,
            self.qf,
        ]
