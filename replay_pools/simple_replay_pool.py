from numpy.random import randint
from collections import OrderedDict
import numpy as np

from replay_pools.base_replay_pool import BaseReplayPool
from utils.replay_pool_utils import path_to_samples, _random_batch_independently

class SimpleReplayPool(BaseReplayPool):

    def __init__(self, env, max_size=1e6):
        self._env = env
        self.max_size = int(max_size)
        o_shape = self._observation_shape = self._env.observation_space.shape
        a_shape = self._action_shape = self._env.action_space.shape
        self.fields = {
            'observations': {
                'shape': o_shape,
                'type': np.float32,
            },
            'next_observations': {
                'shape': o_shape,
                'type': np.float32,
            },
            'actions': {
                'shape': a_shape,
                'type': np.float32,
            },
            'rewards': {
                'shape': (1,),
                'type': np.float32,
            },
            'terminals': {
                'shape': (1,),
                'type': np.float32,
            },
        }
        self.initialize_dataset()

    def initialize_dataset(self):
        self._dataset = {}
        for k, v in self.fields.items():
            self._dataset[k] = np.empty((self.max_size, *v["shape"]), dtype=v["type"])
        self._fields_list = list(self.fields.keys())
        self._size = 0
        self._stop = 0

    def add_paths(self, paths):
        samples = path_to_samples(paths)
        self.add_samples(samples)

    def _preprocess_sample_dict(self, samples):
        return {k: samples[k] for k in self._fields_list}

    def add_samples(self, samples):
        samples = self._preprocess_sample_dict(samples)
        sample_len = len(samples["observations"])
        new_stop = self._stop + sample_len
        if new_stop > self.max_size:
            new_stop = new_stop % self.max_size
            for key, value in samples.items():
                self._dataset[key][self._stop:] = value[:-new_stop]
                self._dataset[key][:new_stop] = value[-new_stop:]
        else:
            for key, value in samples.items():
                self._dataset[key][self._stop:new_stop] = value
        self._stop = new_stop
        # self._size = (self._size + sample_len) % self.max_size
        self._size = min(self._size + sample_len, self.max_size)

    def get_diagnostics(self):
        return OrderedDict([
            ('size', self._size)
        ])

    def random_batch(self, batch_size):
        return _random_batch_independently(self._dataset, batch_size, self._size)

    def get_all_data(self):
        return {k: v[:self._size] for k,v in self._dataset.items()}