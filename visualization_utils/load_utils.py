import torch
import json
import numpy as np
from tqdm import tqdm
import os
import os.path as osp
import matplotlib.pyplot as plt
from math import ceil
from functools import reduce
import pandas as pd

import sys
RL_CODE_PATH = osp.abspath(__file__)
for i in range(2):
    RL_CODE_PATH = osp.dirname(RL_CODE_PATH)
sys.path.insert(0, RL_CODE_PATH)

from policies import *
from value_functions import *
from environments import *
from collectors import *
from visualization_utils.consts import *
from visualization_utils.misc_utils import *
from torch_utils.utils import set_gpu_mode
import torch_utils.utils as ptu

set_gpu_mode(USE_GPU)

def get_configs(exp_dir, variant_file_name="variant.json"):
    variant_path = osp.join(exp_dir, variant_file_name)
    with open(variant_path, "r") as js:
        config_dict = json.load(js)
    return config_dict

def get_snapshot(exp_dir, epoch=None):
    # assert epoch is None
    if epoch is None or epoch=="last": # if not given, then use the last epoch
        epoch = 50 if "InvertedDoublePendulum" in exp_dir else 1000
    elif type(epoch) in [int, np.int32]:
        pass
    else:
        raise NotImplementedError
    snapshot_path = osp.join(exp_dir, f"snapshot/itr_{epoch}.pkl")
    snapshot = torch.load(snapshot_path)
    return snapshot

def get_env_configs(config_dict, e_type="expl", **kwargs):
    for env_dict in config_dict["environments"]:
        env_config_dict, env_class, env_type = env_dict["kwargs"], env_dict["class"], env_dict["name"]
        if env_type[:4] == e_type:
            env_config_dict.update(kwargs)
            return {"config_dict": env_config_dict, "class_name": env_class}
    raise Exception(f"no {e_type} environment found")


def get_expl_policy_configs(config_dict, env, q_func=None):
    for policy_dict in config_dict["policies"]:
        policy_config_dict, policy_class, policy_type = policy_dict["kwargs"], policy_dict["class"], policy_dict["name"]
        if policy_type == "policy":
            policy_config_dict["env"] = env
            return {"config_dict": policy_config_dict, "class_name": policy_class}
    raise Exception("no exploration policy found")

def get_q_func_configs(config_dict, env):
    # for q_func_dict in :
    q_func_dict = config_dict["value_functions"]
    q_func_config_dict, q_func_class = q_func_dict["kwargs"], q_func_dict["class"]
    q_func_config_dict["env"] = env
    return {"config_dict": q_func_config_dict, "class_name": q_func_class}

def get_eval_collector_configs(config_dict, env, policy):
    for collector_dict in config_dict["collectors"]:
        collector_config_dict, collector_class, collector_type = collector_dict["kwargs"], collector_dict["class"], collector_dict["name"]
        if collector_type[:4] == "eval":
            collector_config_dict["env"] = env
            collector_config_dict["policy"] = policy
            return {"config_dict": collector_config_dict, "class_name": collector_class}
    raise Exception("no evaluation collector found")

def snapshot_filter(snapshot, filter_key):
    new_params = {}
    for key, value in snapshot.items():
        split_key = key.split("/")
        if len(split_key) > 1 and split_key[0] == filter_key:
            new_params[split_key[-1]] = value
    return new_params

def load_policy(config_dict, class_name, snapshot, deterministic=True, action_noise=0.):
    policy = load_general_module(config_dict, class_name, snapshot, "policy")
    if deterministic:
        policy = MakeDeterministicWithNoise(policy, action_noise=action_noise) if action_noise > 0 else MakeDeterministic(policy)
    return policy

def load_q_func(config_dict, class_name, snapshot):
    return load_general_module(config_dict, class_name, snapshot, "qf")
    
def load_general_class(config_dict, class_name):
    certain_object = globals()[class_name](**config_dict)
    return certain_object

def load_general_module(config_dict, class_name, snapshot, key):
    module = load_general_class(config_dict, class_name)
    new_snapshot = snapshot_filter(snapshot, key)
    module.load_snapshot(new_snapshot)
    module.to(ptu.device)
    print(f"use the device: {ptu.device}")
    return module

def get_paths_statistic(paths):
    return [np.sum(path["rewards"]) for path in paths]

def get_policy_returns(eval_collector, total_eval_num=TOTAL_EVAL_NUM, max_path_length=MAX_PATH_LENGTH):
    all_returns = []
    n_env = eval_collector._env.n_env if hasattr(eval_collector._env, "n_env") else 1
    total_epochs = total_eval_num // (n_env)
    num_steps_per_epoch = n_env * max_path_length
    for i in tqdm(range(total_epochs)):
        eval_collector.start_epoch(i)
        paths = eval_collector.collect_new_paths(num_steps_per_epoch)
        eval_collector.end_epoch(i)
        returns = get_paths_statistic(paths)
        all_returns.extend(returns)
    return all_returns

def evaluate_policy_returns_with_disturbance_env_and_action(exp_dir, epoch=None, total_eval_num=TOTAL_EVAL_NUM, mass=1., friction=1., action_noise=0.):
    # load config and snapshot
    configs = get_configs(exp_dir)
    snapshot=get_snapshot(exp_dir, epoch)
    # load disturbance env
    expl_env_configs, eval_env_configs = get_env_configs(configs), get_env_configs(configs)
    expl_env, eval_env=load_general_class(**expl_env_configs), load_general_class(**eval_env_configs)
    # change parameters
    if float(mass) != 1.:
        cur_friction = eval_env.env.model.geom_friction
        new_friction = friction * cur_friction
        eval_env.env.model.geom_friction[:] = new_friction
    if float(friction) != 1.:
        cur_mass = eval_env.env.model.body_mass
        new_mass = mass * cur_mass
        eval_env.env.model.body_mass[:] = new_mass
    # load policy
    policy_configs=get_expl_policy_configs(configs, expl_env, q_func=None)
    policy=load_policy(**policy_configs, snapshot=snapshot, action_noise=action_noise)
    # load collector
    eval_collector_configs=get_eval_collector_configs(configs, eval_env, policy)
    eval_collector = load_general_class(**eval_collector_configs)
    returns = get_policy_returns(eval_collector, total_eval_num)
    return returns

def get_heatmap_data(exp_dir, mass_list, friction_list, data_file_name="heatmap", total_eval_num=TOTAL_EVAL_NUM):
    data_dict_save_path = osp.join(exp_dir, f"{data_file_name}_dict.npy")
    epoch_array = LAST_TEN_SIMPLE if "InvertedDoublePendulum" in exp_dir else LAST_TEN
    if osp.isfile(data_dict_save_path):
        data_dict = np.load(data_dict_save_path, allow_pickle=True).item()
        data_dict["xticks"], data_dict["yticks"] = mass_list, friction_list
        origin_data_dict = data_dict["origin_data_dict"]
        data_matrix = np.empty((len(friction_list), len(mass_list)), dtype=np.float32)
        for mass_i, mass in enumerate(mass_list):
            for friction_i, friction in enumerate(friction_list):
                data_key = f"mass-{str(round(mass, 3))}_friction-{str(round(friction, 3))}"
                if data_key in origin_data_dict:
                    value = origin_data_dict[data_key]
                else:
                    returns = []
                    random_array = np.random.choice(epoch_array, size=RANDOM_EPOCH_CHOICE_NUM, replace=False)
                    for epoch in random_array:
                        returns_single_epoch_policy = evaluate_policy_returns_with_disturbance_env_and_action(exp_dir, epoch=epoch, mass=mass, friction=friction, total_eval_num=total_eval_num)
                        returns.extend(returns_single_epoch_policy)
                    value = np.mean(returns)
                    origin_data_dict[data_key] = value
                print(f"{data_key}: {value}", flush=True)
                data_matrix[friction_i, mass_i] = value
    else:
        data_dict = {"xticks": mass_list, "yticks": friction_list}
        data_matrix = np.empty((len(friction_list), len(mass_list)), dtype=np.float32)
        origin_data_dict = {}
        for mass_i, mass in enumerate(mass_list):
            for friction_i, friction in enumerate(friction_list):
                data_key = f"mass-{str(round(mass, 3))}_friction-{str(round(friction, 3))}"
                returns = []
                random_array = np.random.choice(epoch_array, size=RANDOM_EPOCH_CHOICE_NUM, replace=False)
                for epoch in random_array:
                    returns_single_epoch_policy = evaluate_policy_returns_with_disturbance_env_and_action(exp_dir, epoch=epoch, mass=mass, friction=friction, total_eval_num=total_eval_num)
                    returns.extend(returns_single_epoch_policy)
                value = np.mean(returns)
                print(f"{data_key}: {value}", flush=True)
                origin_data_dict[data_key] = value
                data_matrix[friction_i, mass_i] = value
        data_dict["origin_data_dict"] = origin_data_dict
    np.save(data_dict_save_path, data_dict)
    return data_matrix
