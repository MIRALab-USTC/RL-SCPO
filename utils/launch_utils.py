from argparse import ArgumentParser
import os
from os import path as osp
import json
import time
import importlib
import numpy as np
import torch
import random
import datetime
import dateutil.tz
from os.path import join
import warnings

from utils.logger import logger

cur_path = osp.abspath(__file__)
code_path = osp.dirname(osp.dirname(cur_path))
BASE_CONFIG_DIR = osp.join(code_path, "configs")
_LOCAL_LOG_DIR = osp.join(code_path, "data_dir")


def parse_cmd():
    parser = ArgumentParser()
    parser.add_argument("-c", "--config_class", type=str, default="sc-sac")
    parser.add_argument("-n", "--config_name", type=str, default="default")
    parser.add_argument("-p", "--exp_prefix", type=str, default="test")
    parser.add_argument("-i", "--exp_i", type=int, default=0)
    parser.add_argument("-e", "--environment", type=str, default="None")
    args = parser.parse_args()
    return args.config_class, args.config_name, args.exp_prefix, args.exp_i, args.environment


def get_configs(config_class, config_name):
    if config_name[-4:] != "json":
        config_name = config_name + ".json"
    config_path = osp.join(BASE_CONFIG_DIR, config_class, config_name)
    with open(config_path, "r") as config_file:
        configs = json.load(config_file)
    if "base_config_name" in configs:
        base_config_name = configs.pop("base_config_name")
        base_configs = get_configs(config_class, base_config_name)
        base_configs.update(configs)
        configs = base_configs
    return configs


def get_timestamp():
    return time.strftime("%m%d%H%M%S")


def change_configs_from_args(configs, environment_name):
    if environment_name != "None":
        print(f"environment changes to {environment_name}")
        for env_dict in configs["environments"]:
            env_dict["kwargs"]["env_name"] = environment_name

def add_exp_kwargs(configs, seed, exp_id, algo_name, exp_name):
    exp_configs = configs["experiment"]
    exp_configs["seed"] = seed
    exp_configs["exp_id"] = exp_id
    exp_configs["algo_name"] = algo_name
    exp_configs["exp_name"] = exp_name
    env_configs = configs["environments"][0]
    assert env_configs["name"] == "expl_env"
    exp_configs["env_name"] = env_configs["kwargs"]["env_name"]
    


def set_path_and_exp_title(configs, exp_prefix):
    exp_configs = configs["experiment"]
    base_log_dir = exp_configs["base_log_dir"]
    algo_name = exp_configs["algo_name"]
    env_name = exp_configs["env_name"]
    exp_id = exp_configs["exp_id"]
    seed = exp_configs["seed"]

    exp_configs["exp_name"] = exp_name = f'{exp_prefix}_{exp_configs["exp_name"]}'
    exp_configs["exp_title"] = exp_title = f"{exp_id}_{algo_name}_{env_name}_{exp_name}"
    time_str = get_timestamp()
    exp_configs["log_dir"] = log_dir = osp.join(base_log_dir, algo_name, env_name, exp_name, f"{exp_title}_{seed}_{time_str}")

    if osp.exists(log_dir):
        logger.log("WARNING: Log directory already exists {}".format(log_dir))
    os.makedirs(log_dir, exist_ok=True)



def safe_json(data):
    if data is None:
        return True
    elif isinstance(data, (bool, int, float)):
        return True
    elif isinstance(data, (tuple, list)):
        return all(safe_json(x) for x in data)
    elif isinstance(data, dict):
        return all(isinstance(k, str) and safe_json(v) for k, v in data.items())
    return False


def dict_to_safe_json(d):
    """
    Convert each value in the dictionary into a JSON'able primitive.
    :param d:
    :return:
    """
    new_d = {}

    for key, item in d.items():
        if safe_json(item):
            new_d[key] = item
        else:
            if isinstance(item, dict):
                new_d[key] = dict_to_safe_json(item)
            else:
                new_d[key] = str(item)
    return new_d


def setup_logger(
        log_dir,
        exp_title,
        variant,
        text_log_file="debug.log",
        variant_log_file="variant.json",
        tabular_log_file="progress.csv",
        snapshot_mode="all",
        snapshot_gap=1,
        log_tabular_only=False,
        **kwargs):
    """
    Set up logger to have some reasonable default settings.
    Will save log output to log_dir

    exp_name will be auto-generated to be unique.
    If log_dir is specified, then that directory is used as the output dir.

    :param variant:
    :param text_log_file:
    :param variant_log_file:
    :param tabular_log_file:
    :param snapshot_mode:
    :param log_tabular_only:
    :param snapshot_gap:
    :param log_dir:
    :param script_name: If set, save the script name to this.
    :return:
    """
    first_time = True

    if variant is not None:
        logger.log("Variant:")
        logger.log(json.dumps(dict_to_safe_json(variant), indent=2))
        variant_log_path = join(log_dir, variant_log_file)
        logger.log_variant(variant_log_path, variant)

    tabular_log_path = join(log_dir, tabular_log_file)
    text_log_path = join(log_dir, text_log_file)
    logger.add_text_output(text_log_path)

    if first_time:
        logger.add_tabular_output(tabular_log_path)

    else:
        logger._add_output(tabular_log_path, logger._tabular_outputs, logger._tabular_fds, mode='a')
        for tabular_fd in logger._tabular_fds:
            logger._tabular_header_written.add(tabular_fd)

    logger.set_snapshot_dir(log_dir)
    logger.set_snapshot_mode(snapshot_mode)
    logger.set_snapshot_gap(snapshot_gap)
    logger.set_log_tabular_only(log_tabular_only)
    logger.push_prefix(f"[{exp_title}] ")

    logger.log_dir = log_dir
    logger.set_up_dirs_and_files()


def get_item(item_type, item_class_name, kwargs):
    item_path = f"{item_type}"
    module = importlib.import_module(item_path)
    variable = getattr(module, item_class_name)
    return variable(**kwargs)


def set_global_seed(seed):
    if seed is None:
        seed = np.random.randint(0, 4096)
    np.random.seed(seed)    
    random.seed(seed)    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True 
    return seed


def _visit_all_items(config): 
    for item_type, param in config.items():
        if item_type == 'experiment':
            continue
        if isinstance(param, list):
            for p in param:
                item_name = p['name']
                item_kwargs = p.get('kwargs', {})
                yield item_name, item_type, p['class'], item_kwargs
        else:
            item_name = param.get('name', item_type)
            item_kwargs = param.get('kwargs', {})
            yield item_name, item_type, param['class'], item_kwargs


def get_dict_of_items_from_config(config):
    item_dict = {}
    for item_name, item_type, _, _ in _visit_all_items(config):
        item_dict[item_name] = None
    total_instance = 0

    def replace_kwargs(kwargs):
        ready = True
        for k, v in kwargs.items():
            if isinstance(v, str) and v[0] == '$':
                assert v[1:] in item_dict, "Please check your config file. There is no item corresponding to %s"%v
                item = item_dict[v[1:]]
                if item is not None:
                    kwargs[k] = item
                else:
                    ready = False
        return ready 

    while total_instance < len(item_dict):
        for item_name, item_type, item_class_name, item_kwargs in _visit_all_items(config):
            if item_dict[item_name] is not None:
                continue
            if replace_kwargs(item_kwargs):
                item = get_item(item_type, item_class_name, item_kwargs)
                item_dict[item_name] = item
                total_instance += 1
                if total_instance >= len(item_dict):
                    break
    return item_dict
