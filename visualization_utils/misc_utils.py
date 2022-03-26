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
import warnings

import sys
RL_CODE_PATH = osp.abspath(__file__)
for i in range(2):
    RL_CODE_PATH = osp.dirname(RL_CODE_PATH)
sys.path.insert(0, RL_CODE_PATH)

from visualization_utils.consts import *
from utils.launch_utils import get_timestamp


def create_dirs_with_timestamp(base_save_dir, *dirs):
    new_dirs = []
    timestamp = get_timestamp()
    for dir_name in dirs:
        new_path = osp.join(base_save_dir, timestamp, dir_name)
        os.makedirs(new_path, exist_ok=True)
        new_dirs.append(new_path)
    return new_dirs

def get_fig_and_axs(row, column, **kwargs):
    fig, axs = plt.subplots(row, column, figsize=(column * SUB_FIG_SIZE_COL, row * SUB_FIG_SIZE_ROW), **kwargs)
    if column == 1 and row == 1:
        axs = np.array([[axs]])
    elif column == 1:
        axs.shape = (-1,1)
    elif row == 1:
        axs.shape = (1, -1)
    return fig, axs

def get_env_name_simplified(env_name):
    env_name = env_name.split("-")[0]
    if env_name[0].isupper():
        return env_name
    else:
        return env_name.capitalize()

def get_str_ticks(orig_tick):
    new_tick_list = []
    for i, tick in enumerate(orig_tick):
        new_tick = str(round(tick, 3)) if i % 2 == 0 else ""
        new_tick_list.append(new_tick)
    return new_tick_list

def get_color_kwargs(fig_name):
    fig_name = fig_name.lower()
    if "hopper" in fig_name:
        return {"vmin": 1000, "vmax": 3500}
    elif "walker2d" in fig_name:
        return {"vmin": 3000, "vmax": 5000}
    elif "cheetah" in fig_name:
        return {"vmin": 4000, "vmax": 9000} # raw
    elif "pendulum" in fig_name:
        return {"vmin": 3000, "vmax": 10000}
    else:
        warnings.warn(f"unknow env")
        return {}

def get_all_subfiles(par_dir, only_dirs=True):
    sub_files = [osp.join(par_dir, x) for x in os.listdir(par_dir)]
    if only_dirs:
        sub_files = [x for x in sub_files if osp.isdir(x)]
    return sub_files
