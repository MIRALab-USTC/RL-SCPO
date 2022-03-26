import numpy as np
import warnings
import os
import os.path as osp
from seaborn import heatmap

import sys
RL_CODE_PATH = osp.abspath(__file__)
for i in range(2):
    RL_CODE_PATH = osp.dirname(RL_CODE_PATH)
sys.path.insert(0, RL_CODE_PATH)

from visualization_utils.load_utils import *
from torch_utils.utils import set_gpu_mode

set_gpu_mode(USE_GPU)

def plot_heatmap(ax, data, xticks, yticks, title, xlabel="relative mass", ylabel="relative friction"):
    color_kwargs = get_color_kwargs(title)
    im = ax.imshow(data, **color_kwargs, cmap=CMAP)
    xticks, yticks = get_str_ticks(xticks), get_str_ticks(yticks)
    ax.set_xticks(np.arange(len(xticks)))
    ax.set_yticks(np.arange(len(yticks)))
    ax.set_xticklabels(xticks)
    ax.set_yticklabels(yticks)
    ax.set_xlabel(xlabel, fontsize=FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    ax.set_title(title, fontsize=SUB_TITLE_FONTSIZE)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("return", rotation=-90, va="bottom", fontsize=FONTSIZE)
    ax.tick_params(labelsize=LABEL_SIZE)
    cbar.ax.tick_params(labelsize=LABEL_SIZE)
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

def visualize_robustness_performance(
                                     env_name,
                                     algo_name,
                                     dir_name,
                                     disturbance_dict=DISTURBANCE_DICT,
                                     total_eval_num=TOTAL_EVAL_NUM,
                                     fig_save_name="heatmap"):
    fig_save_dir, = create_dirs_with_timestamp("result", "heatmap")
    fig, axs = get_fig_and_axs(row=1, column=1)

    mass_list, friction_list = disturbance_dict[env_name]["mass_list"], disturbance_dict[env_name]["friction_list"]
    fig_name = get_env_name_simplified(env_name) + " " + algo_name
    exp_dirs = get_all_subfiles(dir_name)
    print(f"env : {env_name}, algo: {algo_name}, random_seed_num: {len(exp_dirs)}")
    temp_data_list = []
    for exp_dir in exp_dirs:
        data_matrix = get_heatmap_data(exp_dir, mass_list, friction_list, total_eval_num=total_eval_num)
        temp_data_list.append(data_matrix)
    data_matrix_final = sum(temp_data_list) / len(temp_data_list)
    del data_matrix
    print("!" * 5, f"{fig_name}: min {np.min(data_matrix_final)}, max {np.max(data_matrix_final)}")
    plot_heatmap(ax[0][0], data=data_matrix_final, xticks=mass_list, yticks=friction_list, title=fig_name)
    fig.savefig(osp.join(fig_save_dir, f"{fig_save_name}.png"))
    fig.savefig(osp.join(fig_save_dir, f"{fig_save_name}.pdf"))
    print(f"figure is saved in {fig_save_dir}")
