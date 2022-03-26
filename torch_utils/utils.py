import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn import init

"""
GPU wrappers
"""
device = None
_gpu_available = torch.cuda.is_available()
_use_gpu = False
_gpu_id = 0

def set_gpu_mode(mode, gpu_id=0):
    global _use_gpu, device, _gpu_id, _gpu_available
    _gpu_id = gpu_id
    _use_gpu = mode and _gpu_available
    device = torch.device("cuda:" + str(gpu_id) if _use_gpu else "cpu")
    if _use_gpu:
        set_device(gpu_id)

def gpu_enabled():
    return _use_gpu


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)

def from_numpy(ndarray):
    return torch.from_numpy(ndarray.astype(np.float32)).to(device)

def get_numpy(tensor):
    return tensor.to('cpu').detach().numpy()

nonlinearity_dict = {
    'relu': F.relu,
    'tanh': torch.tanh,
    'identity': lambda x: x,
}

def get_nonlinearity(activation='relu'):
    if activation is None:
        return nonlinearity_dict['identity']
    elif type(activation) is str:
        return nonlinearity_dict[activation]
    elif hasattr(activation, "__call__"):
        return activation
    else:
        raise NotImplementedError 

def tensors(obj, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.tensor(obj, **kwargs, device=torch_device)

def zeros(*sizes, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.zeros(*sizes, **kwargs, device=torch_device)


def ones(*sizes, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.ones(*sizes, **kwargs, device=torch_device)

def normal_distribution(mean, std, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    if type(mean) is np.ndarray:
        mean = from_numpy(mean)
    if type(std) is np.ndarray:
        std = from_numpy(std)
    return Normal(mean.to(device), std.to(device))


def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def np_to_pytorch_batch(np_batch):
    torch_batch = {
        k: _elem_or_tuple_to_variable(x)
        for k, x in np_batch.items()
        if x.dtype != np.dtype('O')  # ignore python object (e.g. dictionaries)
    }
    return torch_batch


def _elem_or_tuple_to_variable(elem_or_tuple):
    if isinstance(elem_or_tuple, tuple):
        return tuple(_elem_or_tuple_to_variable(e) for e in elem_or_tuple)
    return from_numpy(elem_or_tuple)
