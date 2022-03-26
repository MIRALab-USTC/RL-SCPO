import gym
import warnings
import numpy as np
import os
from os import path as osp

def from_offline_env_name_to_env_name(env_name):
    if env_name[0].isupper():
        return env_name
    env_name = env_name.split("-")[0].lower()
    env_name = env_name.capitalize() + "-v2"
    return env_name

obs_mean = 0.
obs_std = 1.
obs_process_func = lambda x: x
inverse_obs_process_func = lambda x: x

def _set_obs_process_func_from_mean_std():
    global obs_process_func, inverse_obs_process_func, obs_mean, obs_std
    obs_process_func = lambda x: (x - obs_mean) / obs_std
    inverse_obs_process_func = lambda x: x * obs_std + obs_mean
    

def get_obs_process_func():
    return obs_process_func, inverse_obs_process_func, obs_mean, obs_std

def get_mean_std_from_data(obs, min_std=1e-3):
    obs_mean = np.mean(obs, axis=0, keepdims=True)
    obs_std = np.std(obs, axis=0, keepdims=True)
    obs_std[obs_std < min_std] = 1.
    return obs_mean, obs_std

def set_obs_process_func_from_data(obs, min_std=1e-3):
    global obs_mean, obs_std
    obs_mean, obs_std = get_mean_std_from_data(obs, min_std=min_std)
    _set_obs_process_func_from_mean_std()

def set_obs_process_func_from_path(npy_path):
    global obs_mean, obs_std
    data = np.load(npy_path, allow_pickle=True).item()
    obs_mean, obs_std = data["obs_mean"], data["obs_std"]
    _set_obs_process_func_from_mean_std()

def make_gym_env(env_name, seed, **kwargs):
    env = gym.make(env_name, **kwargs).env
    env.seed(seed)
    return env

def get_make_fn(env_name, seed, **kwargs):
    def make():
        env = make_gym_env(env_name, seed, **kwargs)
        env.seed(seed)
        return env
    return make

def get_make_fns(env_name, seeds, n_env=1, **kwargs):
    if len(seeds) != n_env:
        raise Exception('the length of the seeds is different from n_env')
    make_fns = [get_make_fn(env_name, seed, **kwargs) for seed in seeds]
    return make_fns


def func0(x):
    return np.piecewise(x, [x<=0, x>0], [lambda y: -(2. * y + 1.)**2 + 1., lambda y: (2. * y - 1.)**2 - 1.])

def func1(x):
    return 0 * x

def func2(x):
    return x ** 2 - 1.
    
def normal_noise_generator(mean=0, std=1, size=int(1e5), shape=()):
    while True:
        normal_noise = np.random.normal(mean, std, (size, *shape), dtype=np.float32)
        for rand_value in normal_noise:
            yield rand_value
    return "done"