from abc import ABC, abstractmethod
import gtimer as gt
from visdom import Visdom

from utils.launch_utils import get_dict_of_items_from_config
from utils.logger import logger
from utils.algorithm_utils import get_epoch_timings
from utils.eval_utils import get_generic_path_information

class BaseAlgorithm(ABC):
    def __init__(self,
                 num_epochs,
                 visdom_port=None,
                 visdom_win=8099,
                 record_video_freq=10,
                 record_parameters_freq=10,
                 need_snapshot_list=None,
                 item_dict_config={}):
        self.num_epochs = num_epochs
        self.item_dict_config = item_dict_config
        self.item_dict = get_dict_of_items_from_config(self.item_dict_config)
        self.__dict__.update(self.item_dict)
        self._need_snapshot = [] if need_snapshot_list is None else need_snapshot_list
        self.record_video_freq = record_video_freq
        self.record_parameters_freq = record_parameters_freq
        self.eval_collector_list = []
        try:
            eval_collector = self.__dict__["eval_collector"]
            self.eval_collector_list.append(eval_collector)
        except KeyError:
            for i in range(0, 10):
                try:
                    eval_collector = self.__dict__[f"eval_collector_{i}"]
                    self.eval_collector_list.append(eval_collector)
                except KeyError:
                    break

        assert len(self.eval_collector_list) > 0

        self.visdom_win = visdom_win
        self.viz = Visdom(port=visdom_port)
        for ith_dict in range(len(self.eval_collector_list)):
            self.viz.line([0], [0], win=visdom_win, opts={"title": visdom_win, "showlegend":True, "xlabel":"step", "ylabel":"average_return"}, name=str(ith_dict))

        # total step
        self.total_step = 0

        # record best performance (epoch, score)
        self.best_performance = (0,0)

    def train(self):
        self._before_train()
        for epoch in gt.timed_for(range(1, self.num_epochs+1)):
            self.train_epoch(epoch)
        self._after_train()

    def train_epoch(self, epoch=None):
        self.start_epoch(epoch)
        self._train_epoch(epoch)
        self.end_epoch(epoch)

    def start_epoch(self, epoch=None):
        for item in self.item_dict.values():
            if hasattr(item, 'start_epoch'):
                item.start_epoch(epoch)
        self._start_epoch(epoch)

    def end_epoch(self, epoch=None):
        for item in self.item_dict.values():
            if hasattr(item, 'end_epoch'):
                item.end_epoch(epoch)
        self._end_epoch(epoch)
        if epoch % self.record_parameters_freq == 0 or (self.num_epochs - epoch) <= 10:
            snapshot = self.get_snapshot()
            logger.save_itr_params(epoch, snapshot)
        self.log_stats(epoch)

    def get_snapshot(self):
        snapshot = {}
        for item_name in self._need_snapshot:
            item = self.item_dict[item_name]
            if hasattr(item, 'get_snapshot'):
                for k, v in item.get_snapshot().items():
                    snapshot[item_name + '/' + k] = v
        return snapshot

    def log_stats(self, epoch):
        logger.log("Epoch {} finished".format(epoch), with_timestamp=True)


        """
        total steps
        """
        logger.record_dict({"epoch":epoch, "total_step": self.total_step})
        """
        Replay Buffer
        """
        if hasattr(self, 'pool'):
            logger.record_dict(
                self.pool.get_diagnostics(),
                prefix='replay_pool/'
            )

        """
        Trainer
        """
        if hasattr(self, 'trainer'):
            train_dict = self.trainer.get_diagnostics()
            logger.record_dict(
                train_dict,
                prefix='trainer/'
            )

        """
        Exploration
        """
        if hasattr(self, 'expl_env') and hasattr(self.expl_env, 'get_diagnostics'):
            logger.record_dict(
                self.expl_env.get_diagnostics(expl_paths),
                prefix='exploration/',
            )

        if hasattr(self, 'expl_collector'):
            expl_paths = self.expl_collector.get_epoch_paths()
            logger.record_dict(
                self.expl_collector.get_diagnostics(),
                prefix='exploration/'
            )
            logger.record_dict(
                get_generic_path_information(expl_paths),
                prefix="exploration/",
            )

        """
        Evaluation
        """
        if hasattr(self, 'eval_env') and hasattr(self.eval_env, 'get_diagnostics'):
            eval_dict = self.eval_env.get_diagnostics(eval_paths)
            logger.record_dict(
                eval_dict,
                prefix='evaluation/',
            )
            
        # if hasattr(self, 'eval_collector'):
        list_of_dict = []
        for ith_eval, eval_collector in enumerate(self.eval_collector_list):
            logger.record_dict(
                eval_collector.get_diagnostics(),
                prefix=f'evaluation_{ith_eval}/',
            )
            eval_paths = eval_collector.get_epoch_paths()
            collector_eval_dict = get_generic_path_information(eval_paths)
            logger.record_dict(
                collector_eval_dict,
                prefix=f"evaluation_{ith_eval}/",
            )
            list_of_dict.append(collector_eval_dict)

        """
        Misc
        """
        # gt.stamp('logging')
        logger.record_dict(get_epoch_timings())
        logger.record_tabular('Epoch', epoch)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)


        """visdom"""
        cur_score_list = []
        for ith_dict, collector_eval_dict in enumerate(list_of_dict):
            self.viz.line([collector_eval_dict["Average Returns"]], [self.total_step], win=self.visdom_win, name=str(ith_dict), update="append")
            cur_score_list.append(collector_eval_dict["Average Returns"])
        
        cur_score = sum(cur_score_list) / len(cur_score_list)
        if cur_score >= self.best_performance[1]:
            self.best_performance = (epoch, cur_score)
            logger.log_best_performance(self.best_performance, cur_score_list)

    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        if hasattr(self, 'trainer'):
            for net in self.trainer.networks:
                net.train(mode)

    def to(self, device):
        for item_name, item in self.item_dict.items():
            if hasattr(item, 'to'):
                item.to(device)

    @abstractmethod
    def _start_epoch(self, epoch):
        raise NotImplementedError


    @abstractmethod
    def _end_epoch(self, epoch):
        raise NotImplementedError


    @abstractmethod
    def _before_train(self):
        raise NotImplementedError

    @abstractmethod
    def _after_train(self):
        raise NotImplementedError

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError
