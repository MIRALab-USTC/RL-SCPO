from collections import OrderedDict
import numpy as np

from utils.misc_utils import combine_item


def cut_path(path, target_length):
    new_path = {}
    for key, value in path.items():
        if type(value) in [dict, OrderedDict]:
            new_path[key] = cut_path(value, target_length)
        else:
            new_path[key] = value[:target_length]
    return new_path


def path_to_samples(paths):
    path_number = len(paths)
    data = paths[0]
    for i in range(1, path_number):
        data = combine_item(data, paths[i])
    return data

def _shuffer_and_random_batch(dataset, batch_size, valid_size, keys=None):
    _batch_index = np.random.permutation(np.arange(valid_size))
    ts = 0 
    while ts < valid_size:
        te = ts + batch_size
        if te + batch_size > valid_size:
            te += batch_size
        yield get_batch(dataset, _batch_index[ts:te], keys=keys)
        ts = te

def _random_batch_independently(dataset, batch_size, valid_size):
    batch_index = np.random.randint(0, valid_size, batch_size)
    return get_batch(dataset, batch_index)


def get_batch(dataset, batch_index):
    batch = {key: value[batch_index] for key, value in dataset.items()}
    return batch
