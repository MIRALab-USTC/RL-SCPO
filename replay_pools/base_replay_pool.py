from abc import ABC, abstractmethod


class BaseReplayPool(ABC):

    @abstractmethod
    def __init__(self, env):
        raise NotImplementedError

    @abstractmethod
    def add_samples(self, samples):
        raise NotImplementedError

    @abstractmethod
    def add_paths(self, paths):
        raise NotImplementedError

    def get_diagnostics(self):
        return {}
