import numpy as np

# robustness exp constants
MAX_PATH_LENGTH = 1000
TOTAL_EVAL_NUM = 8 

# plot constants
N_COLS = 2
SUB_FIG_SIZE_COL = 11.3
SUB_FIG_SIZE_ROW = 8.7
BINS = 8
COLOR_ALPHA = 0.2
COLOR_LIST = ["orangered", "lightseagreen", "cornflowerblue", "orchid", "gray", "darkseagreen", "goldenrod", "darkorange", "mediumorchid", "darkturquoise" ]

FONTSIZE = 23
LEGEND_SIZE = 23
SUB_TITLE_FONTSIZE=26
TITLE_FONTSIZE=28
LABEL_SIZE=20

CMAP = "Reds"
LINE_WIDTHS = .5

WINDOW_LEN = 50

USE_GPU = True

# heatmap
LAST_TEN = np.linspace(991, 1000, 10).astype(np.int32)
LAST_TEN_SIMPLE = np.linspace(41, 50, 10).astype(np.int32)

RANDOM_EPOCH_CHOICE_NUM = 4

DISTURBANCE_DICT = {"Hopper-v2": {"mass_list": np.linspace(0.2, 1.6, 11), "friction_list": np.linspace(0.7, 1.3, 11)},
                    "Walker2d-v2": {"mass_list": np.linspace(0.5, 1.3, 11), "friction_list": np.linspace(0.4, 1.8, 11)},
                    "HalfCheetah-v2": {"mass_list": np.linspace(0.6, 1.3, 11), "friction_list": np.linspace(0.1, 2.7, 11)},
                    "InvertedDoublePendulum-v2": {"mass_list": np.linspace(0.6, 2.6, 11), "friction_list": np.linspace(0.4, 2.4, 11)}}