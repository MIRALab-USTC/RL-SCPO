import torch
from torch import optim, norm
from torch.autograd import grad
import numpy as np
from collections import OrderedDict
from functools import wraps

from trainers.base_trainer import BaseBatchTrainer
from trainers.sac_trainer import SACTrainer
import torch_utils.utils as ptu
from torch_utils.utils import ones, zeros, tensors
from utils.eval_utils import create_stats_ordered_dict

TEMP_ARRAY = tensors(np.full(2, np.nan, dtype=np.float32))
TEMP_NAN = tensors(np.nan)

TD3_POLICY_NOISE = 0.2
TD3_NOISE_CLAMP = 0.5

class PRSACTrainer(SACTrainer):

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

            adversary_update_period=10
    ):
        super(SACTrainer, self).__init__(num_trains_per_batch=num_trains_per_batch)
        self.env = env

        self.qf = qf
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.discount = tensors(discount)

        self.policy_both = policy
        self.protagonist = self.policy = self.policy_both.protagonist
        self.adversary = self.policy_both.adversary

        optimizer_class = getattr(optim, optimizer_class)


        self.use_automatic_entropy_tuning = True
        target_entropy = -np.prod(self.env.action_space.shape).item()  # heuristic value from Tuomas
        self.target_entropy = tensors(target_entropy)
        self.log_alpha = zeros(1, requires_grad=True)
        self.alpha_optimizer = optimizer_class(
            [self.log_alpha],
            lr=log_alpha_lr,)


        self.protagonist_optimizer = self.policy_optimizer = optimizer_class(
            self.protagonist.network.parameters(),
            lr=policy_lr,
        )
        self.adversary_optimizer = optimizer_class(
            self.adversary.network.parameters(),
            lr=policy_lr,
        )
        self.qf_optimizer = optimizer_class(
            self.qf.network.parameters(),
            lr=qf_lr,
        )

        self.epsilon = tensors(self.policy_both.epsilon)
        self.adversary_update_period = adversary_update_period

    def _train_critic(self, obs, actions, rewards, next_obs, terminals, alpha, **kwargs):

        next_action_protagonist, next_policy_info_protagonist = self.protagonist.action(next_obs, return_log_prob=True, fast=True)
        log_prob_next_action_protagonist = next_policy_info_protagonist['log_probs']

        next_action_adversary, _ = self.adversary.action(next_obs, fast=True, noise=TD3_POLICY_NOISE, noise_clamp=TD3_NOISE_CLAMP)

        # get qf network outputs
        q_value_ensemble, _ = self.qf.value(obs, actions)

        # get target soft-q
        _, q_with_regularizer_protagonist = self._get_algo_q_with_regularizer(obs=next_obs, actions=next_action_protagonist, log_prob_action=log_prob_next_action_protagonist.detach(), alpha=alpha.detach(), target=True)
        _, q_info_adversary = self.qf.value(next_obs, next_action_adversary, target=True, fast=True, return_min=True)
        q_with_regularizer_adversary = q_info_adversary["value_min"]

        # get critic loss
        q_target = rewards + (1. - terminals) * self.discount * ( (1-self.epsilon) * q_with_regularizer_protagonist.detach() + self.epsilon * q_with_regularizer_adversary.detach())
        qf_loss = ((q_value_ensemble - q_target.detach()) ** 2).mean()

        # train critic
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()
        return qf_loss, q_value_ensemble, q_target


    def _train_adversary(self, batch):
        rewards, terminals, obs, actions, next_obs = batch['rewards'], batch['terminals'], batch['observations'], batch['actions'], batch['next_observations']
        action_adversary, _ = self.adversary.action(obs)

        _, q_info = self.qf.value(obs, action_adversary, target=False, fast=False, return_min=True)
        q_new_action = q_info["value_min"]

        policy_loss = self.epsilon * q_new_action.mean()

        # train actor
        self.adversary_optimizer.zero_grad()
        policy_loss.backward()
        self.adversary_optimizer.step()
        return policy_loss

    def _train_from_torch_batch(self, batch):
        super()._train_from_torch_batch(batch)
        if self._num_train_steps % self.adversary_update_period == 0:
            self.training_records["adversary_policy_loss"] = self._train_adversary(batch).item()
        else:
            self.training_records["adversary_policy_loss"] = np.nan