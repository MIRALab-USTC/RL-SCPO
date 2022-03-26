from gym.spaces import Box
import numpy as np
import random
from gym import Wrapper
from os import path as osp
import sys

from environments.base_env import BaseEnv
from utils.environment_utils import make_gym_env, set_obs_process_func_from_path, get_obs_process_func, from_offline_env_name_to_env_name


class SimpleEnv(BaseEnv):
    def __init__(self, 
                 env_name,
                 reward_scale=1.0,
                 max_length=np.inf,
                 return_info=False,
                 state_scale_from_data=False,
                 seed=None,
                 **env_kwargs):
        self._set_env(env_name, seed=seed, **env_kwargs)
        self._set_action_scale()
        self._set_reward_scale_and_horizon(reward_scale, max_length)
        self._set_info_process_func(return_info)
        self._set_obs_process_func_from_file(env_name, state_scale_from_data)

    def _set_env(self, env_name, seed=None, **env_kwargs):
        self.env_name = env_name
        self.cur_seed = random.randint(0,65535) if seed is None else seed
        inner_env = make_gym_env(env_name, self.cur_seed, **env_kwargs)
        Wrapper.__init__(self, inner_env)

    def _set_action_scale(self):
        self.low = np.maximum(self.env.action_space.low, -10)
        self.high = np.minimum(self.env.action_space.high, 10)
        ub = np.ones(self.env.action_space.shape)
        self.action_space = Box(-1 * ub, ub)
        self.action_scaler = (self.high - self.low) * 0.5

    def _set_reward_scale_and_horizon(self, reward_scale=1., max_length=np.inf):
        self.reward_scale = reward_scale
        self.max_length = max_length

    def _set_info_process_func(self, return_info=False):
        def info_process_func(info):
            for k, v in info.items():
                info[k] = np.array([[v]])
            return info
        self._info_process_func = info_process_func if return_info else lambda x: {}

    def _set_obs_process_func_from_file(self, env_name, state_scale_from_data=False):
        if state_scale_from_data:
            obs_mean_std_dir = osp.join(sys.path[0], "data")
            obs_mean_std_path = osp.join(obs_mean_std_dir, f"{from_offline_env_name_to_env_name(env_name)}.npy")
            print(f"state scale from {obs_mean_std_path}")
            assert osp.isfile(obs_mean_std_path)
            set_obs_process_func_from_path(obs_mean_std_path)
        self._obs_process_func, self._inverse_obs_process_func, self._obs_mean, self._obs_std = get_obs_process_func()

    def reset(self):
        self.cur_step_id = 0
        return self._obs_process_func(self.env.reset()[np.newaxis])

    def _action_process(self, action):
        return self.low + (action.clip(-1.0, 1.0) + 1.0) * self.action_scaler

    def _outputs_process(self, o, r, d, info):
        """process o, r, d, info"""

        # process d
        if self.cur_step_id < self.max_length:
            d = np.float32(d)
        else:
            d = 1.0

        # add one dimension at first axis for all data
        return self._obs_process_func(o[np.newaxis]), np.array([[r]]), np.array([[d]]), self._info_process_func(info)

    def step(self, action):
        self.cur_step_id += 1

        # action scale
        action = self._action_process(action)

        # take step and get original returns
        o, r, d, info = self.env.step(action)

        # return processed outputs
        return self._outputs_process(o, r, d, info)
