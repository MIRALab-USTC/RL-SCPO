import random
import numpy as np
from gym.spaces import Box
from gym import Wrapper
from gym.vector import SyncVectorEnv, AsyncVectorEnv

from environments.simple_env import SimpleEnv
from utils.environment_utils import get_make_fns, from_offline_env_name_to_env_name


class NormalizedVectorEnv(SimpleEnv):
    def __init__(self, 
                 env_name,
                 n_env=1,
                 asynchronous=True,

                 reward_scale=1.0,
                 max_length=np.inf,
                 return_info=False,
                 state_scale_from_data=False,

                 seed=None,
                 **vector_env_kwargs):
        env_name = from_offline_env_name_to_env_name(env_name)
        self._set_vector_env(env_name,
                             n_env=n_env,
                             asynchronous=asynchronous,
                             seed=seed,
                             **vector_env_kwargs)
        super().__init__(env_name,
                         reward_scale=reward_scale,
                         max_length=max_length,
                         return_info=return_info,
                         state_scale_from_data=state_scale_from_data)

    def _set_vector_env(self, env_name, n_env=1, asynchronous=True, seed=None, **vector_env_kwargs):
        self.env_name = env_name
        self.n_env = n_env
        if seed is None:
            self.cur_seeds = [random.randint(0, 65535) for i in range(n_env)]
        else:
            raise NotImplementedError
        self.make_fns = get_make_fns(env_name, self.cur_seeds, n_env, **vector_env_kwargs)
        if asynchronous:
            inner_env = AsyncVectorEnv(self.make_fns)
        else:
            inner_env = SyncVectorEnv(self.make_fns)
        Wrapper.__init__(self, inner_env)
        self.observation_space = self.env.single_observation_space

    def _set_env(self, env_name, seed=None, **env_kwargs):
        pass

    def _set_action_scale(self):
        self.low = np.maximum(self.env.single_action_space.low, -10)
        self.high = np.minimum(self.env.single_action_space.high, 10)
        ub = np.ones(self.env.single_action_space.shape)
        self.action_space = Box(-1 * ub, ub)
        self.action_scaler = (self.high - self.low) * 0.5

    def _set_info_process_func(self, return_info=False):
        def info_process_func(info):
            raise NotImplementedError
        self._info_process_func = info_process_func if return_info else lambda x: {}

    def reset(self):
        self.cur_step_id = 0
        return self._obs_process_func(self.env.reset())

    def _action_process(self, action):
        action_processed = self.low + (action.clip(-1.0, 1.0) + 1.0) * self.action_scaler
        # stack the actions if there is only one action
        if len(action_processed.shape) == len(self.action_space.shape):
            action_processed = np.stack([action_processed] * self.n_env)
        return action_processed

    def _outputs_process(self, o, r, d, info):
        """process o, r, d, info"""

        # process d
        if self.cur_step_id < self.max_length:
            d = d.reshape(self.n_env,1).astype(np.float)
        else:
            d = np.ones((self.n_env,1), dtype=np.float)

        # add one dimension at first axis for all data
        return self._obs_process_func(o), r.reshape(self.n_env,1), d, self._info_process_func(info)
