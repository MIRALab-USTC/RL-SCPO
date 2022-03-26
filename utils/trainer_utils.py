import torch
from torch.nn.functional import softmin
from math import gamma, pi, sqrt
from torch_utils.utils import ones, tensors
import numpy as np

MIN_VALUE = 1e-6

def get_weight_no_grad(ensembled_v):
    with torch.no_grad():
        v1, v2 = torch.chunk(ensembled_v, 2, dim=-1)
        delta_v = v1 - v2
        abs_delta_v = torch.abs(delta_v)
        shape = abs_delta_v.shape
        abs_delta_v = abs_delta_v.reshape(-1)
        weight = softmin(abs_delta_v)
        weight = weight.reshape(*shape)
    return weight

def get_radius_from_noise_epsilon(dimension, noise_epsilon=0.1):
    return 2 * noise_epsilon * gamma(dimension / 2 + 1)**(1 / dimension) / sqrt(pi)

def get_uniform_weights(size):
    return ones(*size)

def clear_grad(value):
    if value.grad is not None:
        value.grad.requires_grad_(False)
        value.grad.zero_()

def psgd_optimizer(value, gradient, lr=1e-3, projection=np.inf):
    # gradient ascent
    gradient_max = torch.max(gradient, dim=-1, keepdim=True)[0].clamp_(min=MIN_VALUE)
    true_lr = lr / gradient_max
    delta = true_lr * gradient
    value.detach_()
    value += delta
    value.clamp_(-projection, projection)
    value.requires_grad_()
    # clear gradient
    clear_grad(value)