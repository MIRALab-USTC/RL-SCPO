from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import contextmanager

import torch

from torch_utils.utils import from_numpy

class BaseTrainer(ABC):

    def __init__(self):
        self._num_train_steps = 0
        self.temp_info = {}
        self.diagnostics = OrderedDict()
        self.training_records = OrderedDict()
        self._need_to_update_diagnostics = True

    def start_epoch(self, epoch):
        self.epoch = epoch

    def end_epoch(self, epoch):
        self._need_to_update_diagnostics = True

    def get_snapshot(self):
        return {}

    def get_diagnostics(self):
        return self.diagnostics

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    @contextmanager
    def with_train_mode_true(self):
        self.training_mode(True)
        yield
        self.training_mode(False)

    def _preprocess_data(self, np_dict):
        tensor_dict = {k:from_numpy(v) for k,v in np_dict.items()}
        return tensor_dict

    # >>> need to be implemented
    @abstractmethod
    def networks(self):
        raise NotImplementedError

    @abstractmethod
    def train(self, data):
        pass

    @abstractmethod
    def _update_diagnostics(self):
        self.diagnostics.update(self.training_records)
    # <<< need to be implemented

class BaseBatchTrainer(BaseTrainer):

    def __init__(self, num_trains_per_batch=1):
        super().__init__()
        self.num_trains_per_batch = num_trains_per_batch

    def train(self, np_data):
        self._num_train_steps += 1
        processed_data = self._preprocess_data(np_data)
        with self.with_train_mode_true():
            for _ in range(self.num_trains_per_batch):
                self._train_from_torch_batch(processed_data)

        if self._need_to_update_diagnostics:
            self._need_to_update_diagnostics = False
            self._update_diagnostics()

        return self.training_records

    @abstractmethod
    def _train_from_torch_batch(self, batch):
        raise NotImplementedError


BaseTrainer.register(BaseBatchTrainer)
