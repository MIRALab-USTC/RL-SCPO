from os import path as osp
from copy import deepcopy
import sys

import torch_utils.utils as ptu
from utils.logger import logger
from utils.launch_utils import parse_cmd, get_configs, set_global_seed, get_item, setup_logger, set_path_and_exp_title, change_configs_from_args, add_exp_kwargs

cur_path = osp.abspath(__file__)
code_path = osp.dirname(cur_path)
sys.path.insert(0, code_path)


def run_experiments():
    config_class, config_name, exp_prefix, ith_exp, environment_name= parse_cmd()
    configs = get_configs(config_class, config_name)
    change_configs_from_args(configs, environment_name)
    experiment_kwargs = configs.get("experiment")
    seed = experiment_kwargs.get("seed", None)
    seed = set_global_seed(seed)
    add_exp_kwargs(configs, seed, ith_exp, config_class, config_name.split('/')[-1])
    set_path_and_exp_title(configs, exp_prefix)

    logger.reset()
    setup_logger(variant=configs, **experiment_kwargs)
    use_gpu = experiment_kwargs.get('use_gpu', True)
    ptu.set_gpu_mode(use_gpu)

    configs.pop("experiment")
    algo = configs.pop("algorithm")
    algo_kwargs = algo["kwargs"]
    algo_kwargs['item_dict_config'] = configs
    algo_kwargs["visdom_port"] = experiment_kwargs["visdom_port"]
    algo_kwargs["visdom_win"] = experiment_kwargs["exp_title"]
    algo_kwargs["silent"] = experiment_kwargs.get("silent", True)
    algo = get_item("algorithms", algo["class"], algo_kwargs)

    algo.to(ptu.device)
    algo.train()


if __name__ == "__main__":
    run_experiments()
