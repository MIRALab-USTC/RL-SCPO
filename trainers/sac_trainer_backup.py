import torch
from torch import optim
import numpy as np
from collections import OrderedDict

from trainers.base_trainer import BaseBatchTrainer
import torch_utils.utils as ptu
from torch_utils.utils import ones, zeros, tensors
from utils.eval_utils import create_stats_ordered_dict


class SACTrainer(BaseBatchTrainer):

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
    ):
        super().__init__(plotter)
        self.env = env
        self.policy = policy
        self.qf = qf
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.discount = tensors(discount)

        optimizer_class = getattr(optim, optimizer_class)
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy is None:
                target_entropy = -np.prod(self.env.action_space.shape).item()  # heuristic value from Tuomas
            self.target_entropy = tensors(target_entropy)
            self.log_alpha = zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=log_alpha_lr,
            )
        else:
            self.alpha_if_not_automatic = tensors(alpha_if_not_automatic)
        self.policy_optimizer = optimizer_class(
            self.policy.network.parameters(),
            lr=policy_lr,
        )
        self.qf_optimizer = optimizer_class(
            self.qf.network.parameters(),
            lr=qf_lr,
        )

    def _update_target_networks(self):
        if self._num_train_steps % self.target_update_period == 0:
            self.qf.update_target(self.soft_target_tau)

    def _train_temperature_alpha(self, rewards, terminals, obs, actions, next_obs, new_action, log_prob_new_action, **kwargs):
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob_new_action.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.detach().exp()
        else:
            alpha_loss = 0
            alpha = self.alpha_if_not_automatic
        return alpha, alpha_loss.detach()

    # def _get_q_func(self, target):
    #     return self.qf.value_target_fast if target else self.qf.value

    def _get_algo_q_with_regularizer(self, obs, actions, log_prob_action, alpha, target=False, fast=None):
        """this function returns soft q for sac_algorithm"""
        # q_func = self._get_q_func(target=target)
        if fast is None:
            fast = target
        _, q_info = self.qf.value(obs, actions, target=target, fast=fast, return_min=True)
        q_min = q_info["value_min"]

        q_with_regularizer = q_min - alpha * log_prob_action
        return q_min, q_with_regularizer

    def _train_critic(self, rewards, terminals, obs, actions, next_obs, next_action, log_prob_next_action, alpha, **kwargs):
        # Make sure policy accounts for squashing functions like tanh correctly!
        
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

    def _train_actor(self, rewards, terminals, obs, actions, next_obs, new_action, log_prob_new_action, alpha, **kwargs):
        # get actor loss
        q_new_action, q_with_regularizer = self._get_algo_q_with_regularizer(obs=obs, actions=new_action, log_prob_action=log_prob_new_action, alpha=alpha)
        policy_loss = (-q_with_regularizer).mean()

        # train actor
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        return policy_loss, q_new_action

    def _get_diagnostics_dict(self):
        q_new_action, log_prob_new_action, policy_loss, qf_loss = self.temp_info_dict["q_new_action"], self.temp_info_dict["log_prob_new_action"], self.temp_info_dict["policy_loss"], self.temp_info_dict["qf_loss"]
        with torch.no_grad():
            average_entropy = -log_prob_new_action.mean()
            policy_q_loss = 0 - q_new_action.mean()

        diagnostics = OrderedDict()
        diagnostics['Policy Loss'] = np.mean(ptu.get_numpy(policy_loss))
        diagnostics['Policy Q Loss'] = np.mean(ptu.get_numpy(policy_q_loss))
        diagnostics['Averaged Entropy'] = np.mean(ptu.get_numpy(average_entropy))
        diagnostics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
        # diagnostics['Terminals'] = np.mean(ptu.get_numpy(terminals))
        # diagnostics['Rewards'] = np.mean(ptu.get_numpy(rewards))
        if self.use_automatic_entropy_tuning:
            diagnostics['Alpha'] = self.temp_info_dict["alpha"].item()
            diagnostics['Alpha Loss'] = self.temp_info_dict["alpha_loss"].item()
        return diagnostics

    def _update_eval_statistics(self, diagnostics):
        q_value_ensemble, q_target, log_prob_new_action = self.temp_info_dict["q_value_ensemble"], self.temp_info_dict["q_target"], self.temp_info_dict["log_prob_new_action"]
        """
        Eval should set this to None.
        This way, these statistics are only computed for one batch.
        """
        self.eval_statistics.update(create_stats_ordered_dict(
            'Q1 Predictions',
            ptu.get_numpy(q_value_ensemble[:, 0]),
        ))
        self.eval_statistics.update(create_stats_ordered_dict(
            'Q2 Predictions',
            ptu.get_numpy(q_value_ensemble[:, 1]),
        ))
        self.eval_statistics.update(create_stats_ordered_dict(
            'Q Targets',
            ptu.get_numpy(q_target),
        ))
        self.eval_statistics.update(create_stats_ordered_dict(
            'Log Pis',
            ptu.get_numpy(log_prob_new_action),
        ))
        self.eval_statistics.update(diagnostics)

    @torch.no_grad()
    def _get_statics(self):
        diagnostics = self._get_diagnostics_dict()
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            self._update_eval_statistics(diagnostics)
        return diagnostics

    def _set_obs(self, obs, next_obs):
        return obs, next_obs

    def _train_from_torch_batch(self, batch, **kwargs):
        rewards, terminals, obs, actions, next_obs = batch['rewards'], batch['terminals'], batch['observations'], batch['actions'], batch['next_observations']
        obs, next_obs = self._set_obs(obs, next_obs)
        train_dataset = {"rewards": rewards, "terminals": terminals, "obs": obs, "actions": actions, "next_obs": next_obs}

        """
        Alpha
        """
        new_action, action_info = self.policy.action(obs, return_log_prob=True)
        log_prob_new_action = action_info["log_probs"]
        self.temp_info_dict["alpha"], self.temp_info_dict["alpha_loss"] = self._train_temperature_alpha(**train_dataset, new_action=new_action, log_prob_new_action=log_prob_new_action)
        self.temp_info_dict["log_prob_new_action"] = log_prob_new_action

        """
        QF 
        """
        next_action, next_policy_info = self.policy.action(next_obs, return_log_prob=True)
        log_prob_next_action = next_policy_info['log_probs']
        self.temp_info_dict["qf_loss"], self.temp_info_dict["q_value_ensemble"], self.temp_info_dict["q_target"] = self._train_critic(**train_dataset, next_action=next_action, log_prob_next_action=log_prob_next_action, alpha=self.temp_info_dict["alpha"])

        """
        Soft Updates
        """
        self._update_target_networks()

        """
        policy
        """
        self.temp_info_dict["policy_loss"], self.temp_info_dict["q_new_action"] = self._train_actor(**train_dataset, new_action=new_action, log_prob_new_action=log_prob_new_action, alpha=self.temp_info_dict["alpha"])

        """
        Compute some statistics for eval
        """
        return self._get_statics()


    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.qf,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf
        )
