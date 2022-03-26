import gtimer as gt
from functools import partial
from warnings import warn

from algorithms.base_algorithm import BaseAlgorithm
from utils.logger import logger
from utils.progress import Progress, Silent
from utils.collector_utils import rollout
from utils.misc_utils import format_for_process

class BatchAlgorithm(BaseAlgorithm):

    def __init__(
                self,
                num_epochs,
                batch_size,

                num_train_loops_per_epoch=1000,
                num_expl_steps_per_train_loop=1,
                num_trains_per_train_loop=1,
                num_trains_per_batch=1,
                min_num_steps_before_training=0,

                num_eval_steps_per_epoch=8000,
                max_path_length=1000,

                silent=False,
                record_video_freq=10,
                record_parameters_freq=10,
                item_dict_config={},
                need_snapshot_list=None,

                visdom_port=8099,
                visdom_win=None,
                save_batch=False):
        super().__init__(num_epochs=num_epochs,
                         visdom_port=visdom_port,
                         visdom_win=visdom_win,
                         record_video_freq=record_video_freq,
                         record_parameters_freq=record_parameters_freq,
                         item_dict_config=item_dict_config,
                         need_snapshot_list=need_snapshot_list)
        # self._need_snapshot.append('trainer')
        self.batch_size = batch_size
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_trains_per_batch = num_trains_per_batch
        self.min_num_steps_before_training = min_num_steps_before_training
        self.max_path_length = max_path_length

        self.progress_class = Silent if silent else Progress
        
        # save batch
        self.save_batch = save_batch

        # others
        self._get_functions()


    def _get_functions(self):
        self._get_collector_func()
        self._get_video_func()
        self._get_video_policy()

    def _get_collector_func(self):
        if hasattr(self.expl_collector, 'collect_new_paths'):
            self.collect_data_func = self.expl_collector.collect_new_paths
            self.add_data_func = self.pool.add_paths
        elif hasattr(self.expl_collector, 'collect_new_steps'):
            self.collect_data_func = self.expl_collector.collect_new_steps
            self.add_data_func = self.pool.add_samples

    def _get_video_policy(self):
        eval_policy_str_list = ["eval_policy"] + [f"eval_policy_{i}" for i in range(5)]
        for eval_policy_str in eval_policy_str_list:
            if hasattr(self, eval_policy_str):
                self.video_policy = getattr(self, eval_policy_str)
                return
        raise Exception("no eval_policy for video")

    def _get_video_func(self):
        # if no video environment, then return empty function
        self.video_env_list=[]
        video_envs_name_list = ["video_env"] + [f"video_env_{i}" for i in range(5)]
        for video_env_name in video_envs_name_list:
            if hasattr(self, video_env_name):
                self.video_env_list.append(getattr(self, video_env_name))
        if len(self.video_env_list) == 0:
            warn("no video environment! please check the config file!")
            self.video_func = lambda epoch, max_path_length : None
        else:
            self.video_func = self._collect_video_from_video_env

    def _collect_video_from_video_env(self, epoch, max_path_length=1000):
        for env in self.video_env_list:
            env.set_video_name("epoch{}".format(epoch))
            logger.log("---------------start: rollout to save video---------------")
            rollout(env, self.video_policy, max_path_length=max_path_length, use_tqdm=True)
            logger.log("---------------end: rollout to save video---------------")

    def _sample(self, num_steps):
        if num_steps > 0:
            data = self.collect_data_func(num_steps, self.max_path_length, True)
            self.add_data_func(data)

    def _before_train(self):
        self.start_epoch(-1)
        if hasattr(self, 'init_expl_policy'):
            with self.expl_collector.with_policy(self.init_expl_policy):
                self._sample(self.min_num_steps_before_training)
        else:
            self._sample(self.min_num_steps_before_training)
        self.total_step += self.min_num_steps_before_training

        for item in self.item_dict.values():
            if hasattr(item, 'end_epoch'):
                item.end_epoch(-1)
        self._end_epoch(-1)


    def _after_train(self):
        if self.save_batch:
            dataset = self.pool.get_all_data()
            logger.log_dataset(dataset)

    def _start_epoch(self, epoch):
        pass

    def _end_epoch(self, epoch):
        if epoch % self.record_video_freq == 0:
            self.video_func(epoch, self.max_path_length)

    def _train_epoch(self, epoch):
        progress = self.progress_class(self.num_train_loops_per_epoch * self.num_trains_per_train_loop)
        for _ in range(self.num_train_loops_per_epoch):
            self._sample(self.num_expl_steps_per_train_loop)
            # gt.stamp('exploration sampling', unique=False)

            self.training_mode(True)
            for _ in range(self.num_trains_per_train_loop):
                progress.update()
                train_data = self.pool.random_batch(self.batch_size)
                params = self.trainer.train(train_data)
                progress.set_description(format_for_process(params))
            # gt.stamp('training', unique=False)
            self.training_mode(False)
        self.total_step += self.num_train_loops_per_epoch * self.num_expl_steps_per_train_loop
        self._eval()
        progress.close()

    def _eval(self):
        if type(self.num_eval_steps_per_epoch) is list:
            for ith_num, eval_collector in enumerate(self.eval_collector_list):
                eval_collector.collect_new_paths(
                    self.num_eval_steps_per_epoch[ith_num],
                    self.max_path_length,
                    discard_incomplete_paths=True,)
        else:
            for eval_collector in self.eval_collector_list:
                eval_collector.collect_new_paths(
                    self.num_eval_steps_per_epoch,
                    self.max_path_length,
                    discard_incomplete_paths=True,)