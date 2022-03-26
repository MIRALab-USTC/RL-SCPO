from abc import ABC, abstractmethod


class BaseCollector(ABC):

    def __init__(
            self,
            env,
            policy,
            max_num_epoch_paths_saved=None,
            render=False,
            render_kwargs={},
            **collector_kwargs):
        self._env = env
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._render = render
        self._render_kwargs = render_kwargs
        self._action_kwargs = collector_kwargs.get("action_kwargs", {})
        self._step_kwargs = collector_kwargs.get("step_kwargs", {})

    @abstractmethod
    def start_epoch(self, epoch=None):
        raise NotImplementedError

    @abstractmethod
    def end_epoch(self, epoch=None):
        raise NotImplementedError

    @abstractmethod
    def get_diagnostics(self):
        raise NotImplementedError

    @abstractmethod
    def get_epoch_paths(self):
        raise NotImplementedError


class BasePathCollector(BaseCollector):

    @abstractmethod
    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
    ):
        raise NotImplementedError


class BaseStepCollector(BaseCollector):

    @abstractmethod
    def collect_new_steps(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
    ):
        raise NotImplementedError

BaseCollector.register(BaseStepCollector)
BaseCollector.register(BasePathCollector)
