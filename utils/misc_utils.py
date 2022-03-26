import numpy as np
from collections import OrderedDict
from contextlib import contextmanager


def combine_item(item, cur_item):
    if type(item) is list:
        assert type(cur_item) is list
        new_item = item + cur_item
    elif type(item) is np.ndarray:
        assert type(cur_item) is np.ndarray
        new_item = np.concatenate([item, cur_item])
    elif type(item) is dict or type(item) is OrderedDict:
        new_item = OrderedDict()
        for key in item:
            assert key in cur_item
            new_item[key] = combine_item(item[key], cur_item[key])
    else:
        raise NotImplementedError
    return new_item


def combine_items(data, cur_data):
    new_data = [combine_item(item, cur_item) for item, cur_item in zip(data, cur_data)]
    return tuple(new_data)

def format_for_process(params):
    new_params = []
    for k, v in params.items():
        k = k.rjust(18)
        v = '{:12}'.format(v)
        new_params.append([k, v])
    return new_params

@contextmanager
def nullcontext(enter_result=None):
    """empty context manager for python<=3.6"""
    yield enter_result
