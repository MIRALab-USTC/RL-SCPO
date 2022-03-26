import numpy as np

LOG_STD_MIN = -20
LOG_STD_MAX = 2

ACTION_MIN = -1.
ACTION_MAX = 1.
SAFE_ACTION_MIN = ACTION_MIN + 1e-6
SAFE_ACTION_MAX = ACTION_MAX - 1e-6

def uniform_noise_generator(min_value=0., max_value=1., size=int(1e5), shape=()):
    while True:
        uniform_noise = np.random.uniform(min_value, max_value, (size, *shape)).astype(np.float32)
        for uniform_value in uniform_noise:
            yield uniform_value
    return "done"

def normal_noise_generator(sigma, shape, size=int(1e5)):
    while True:
        normal_noise_array = np.random.normal(loc=0, scale=sigma, size=(size, *shape))
        for normal_noise in normal_noise_array:
            yield normal_noise
    return "done"

def rand_int_generator(max_value, min_value=0, size=int(1e4), shape=()):
    while True:
        rand_int_array = np.random.randint(low=min_value, high=max_value, size=(size, *shape)).astype(np.int64)
        for rand_int in rand_int_array:
            yield rand_int
    return "done"