import sys
import os
import os.path as osp
from argparse import ArgumentParser

RL_CODE_PATH = osp.abspath(__file__)
RL_CODE_PATH = osp.dirname(RL_CODE_PATH)
sys.path.insert(0, RL_CODE_PATH)

from visualization_utils.plot_utils import *


"""heatmap"""
parser = ArgumentParser()
parser.add_argument("env_name", type=str)
parser.add_argument("algo_name", type=str)
parser.add_argument("save_dir", type=str)

args = parser.parse_args()
env_name, algo_name, save_dir = args.env_name, args.algo_name, args.save_dir

visualize_robustness_performance(env_name, algo_name, save_dir)
