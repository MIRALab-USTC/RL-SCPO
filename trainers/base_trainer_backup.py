from abc import ABC, abstractmethod
from collections import OrderedDict

from torch_utils.utils import np_to_pytorch_batch


class BaseTrainer(ABC):

    @abstractmethod
    def train(self, data):
        pass

    def start_epoch(self, epoch):
        pass

    def end_epoch(self, epoch):
        pass

    def get_snapshot(self):
        return {}

    def get_diagnostics(self):
        return {}


class BaseBatchTrainer(BaseTrainer):

    def __init__(self, ploter):
        self.ploter = ploter
        self._num_train_steps = 0
        self.eval_statistics = OrderedDict()
        self.temp_info_dict = {}
        self._need_to_update_eval_statistics = True

    def _get_processed_batch(self, np_data):
        return np_to_pytorch_batch(np_data)

    def train(self, np_data, **kwargs):
        self._num_train_steps += 1
        processed_data = self._get_processed_batch(np_data)
        return self._train_from_torch_batch(processed_data, **kwargs)

    def get_diagnostics(self):
        return OrderedDict([
            ('num train calls', self._num_train_steps)])

    @abstractmethod
    def _train_from_torch_batch(self, batch):
        raise NotImplementedError


BaseTrainer.register(BaseBatchTrainer)
