import abc
from gym import Wrapper


class BaseEnv(Wrapper, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __init__(self, env_name):
        raise NotImplementedError
    
    @property
    def horizon(self):
        return self.max_length

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError

    @abc.abstractmethod
    def step(self):
        raise NotImplementedError
