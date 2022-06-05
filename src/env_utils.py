import parl
import carla
import gym
import gym_carla
import numpy as np
from parl.utils import csv_logger, logger, tensorboard
from parl.env.continuous_wrappers import ActionMappingWrapper
from parl.utils import CSVLogger
from collections import deque

csv_logger = CSVLogger("Episode_rewad.csv")


class ParallelEnv(object):
    def __init__(self, env_name, xparl_addr, train_envs_params, use_agc=True):
        parl.connect(xparl_addr)
        self.env_list = [
            CarlaRemoteEnv(env_name=env_name, params=params)
            for params in train_envs_params
        ]
        self.env_num = len(self.env_list)
        self.episode_reward_list = [0] * self.env_num
        self.episode_steps_list = [0] * self.env_num
        self._max_episode_steps = train_envs_params[0]["max_time_episode"]
        self.total_steps = 0
        self.n_episodes_completed = 0
        self.n_tasks = len(self.env_list[0].tasks().get())
        self.valid_tasks = np.arange(0, self.n_tasks, 1).astype("int").tolist()
        self.empirical_task_means = np.zeros(self.n_tasks)
        self.task_completed_counts = np.ones(self.n_tasks)
        self.task_success_hist_Qs = [deque(maxlen=10)] * self.n_tasks
        self.use_agc = use_agc

    def reset(self):
        obs_list = [env.reset() for env in self.env_list]
        obs_list = [obs.get() for obs in obs_list]
        self.obs_list = np.array(obs_list)
        return self.obs_list

    def step(self, action_list):
        return_list = [
            self.env_list[i].step(action_list[i]) for i in range(self.env_num)
        ]
        return_list = [return_.get() for return_ in return_list]
        return_list = np.array(return_list, dtype=object)
        self.next_obs_list = return_list[:, 0]
        self.reward_list = return_list[:, 1]
        self.done_list = return_list[:, 2]
        self.info_list = return_list[:, 3]
        return self.next_obs_list, self.reward_list, self.done_list, self.info_list

    def get_obs(self):
        for i in range(self.env_num):
            self.total_steps += 1
            self.episode_steps_list[i] += 1
            self.episode_reward_list[i] += self.reward_list[i]

            self.obs_list[i] = self.next_obs_list[i]
            if (
                self.done_list[i]
                or self.episode_steps_list[i] >= self._max_episode_steps
            ):
                tensorboard.add_scalar(
                    "train/episode_reward_env{}".format(i),
                    self.episode_reward_list[i],
                    self.total_steps,
                )
                logger.info(
                    "Train env {} done, Reward: {}".format(
                        i, self.episode_reward_list[i]
                    )
                )
                if self.use_agc:
                    self.n_episodes_completed += 1
                    task_i_completed: int = self.env_list[i].curr_task().get()
                    prev_mean = self.empirical_task_means[task_i_completed]
                    prev_counts = self.task_completed_counts[task_i_completed]
                    r = self.episode_reward_list[i]
                    self.empirical_task_means[task_i_completed] = (
                        prev_counts * prev_mean + r
                    ) / (prev_counts + 1)
                    self.task_completed_counts[task_i_completed] += 1
                    success_hist = self.task_success_hist_Qs[task_i_completed]
                    if self.env_list[i].is_success().get():
                        success_hist.append(1)
                    else:
                        success_hist.append(0)

                    avg_task_success = sum(success_hist) / len(success_hist)
                    if len(success_hist) >= 10 and avg_task_success >= 0.9:
                        if task_i_completed in self.valid_tasks:
                            self.valid_tasks.remove(task_i_completed)
                            self.empirical_task_means[task_i_completed] = -np.inf
                            self.n_episodes_completed = 1
                    epsilon = 1 / self.n_episodes_completed
                    if np.random.rand(1) < epsilon:
                        next_task_i = int(np.random.choice(self.valid_tasks))
                    else:
                        next_task_i = self.empirical_task_means.argmax(0)
                else:
                    next_task_i = int(np.random.choice(self.valid_tasks))
                tensorboard.add_scalar(
                    "train_meta/task_selection", next_task_i, self.total_steps
                )
                self.episode_steps_list[i] = 0
                self.episode_reward_list[i] = 0
                obs_list_i = self.env_list[i].reset(next_task_i)
                self.obs_list[i] = obs_list_i.get()
                self.obs_list[i] = np.array(self.obs_list[i])
        return self.obs_list


class LocalEnv(object):
    def __init__(self, env_name, params):
        self.env = gym.make(env_name, params=params)
        self.env = ActionMappingWrapper(self.env)
        self._max_episode_steps = int(params["max_time_episode"])
        self.obs_dim = self.env.state_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

    def reset(self, ped_task_i=0):
        self.env.set_ped_task_i(ped_task_i)
        obs, _ = self.env.reset()
        return obs

    def step(self, action):
        return self.env.step(action)


@parl.remote_class(wait=False)
class CarlaRemoteEnv(object):
    def __init__(self, env_name, params):
        class ActionSpace(object):
            def __init__(
                self, action_space=None, low=None, high=None, shape=None, n=None
            ):
                self.action_space = action_space
                self.low = low
                self.high = high
                self.shape = shape
                self.n = n

            def sample(self):
                return self.action_space.sample()

        self.env = gym.make(env_name, params=params)
        # self.env = gym.make('carla-v0')
        self.env = ActionMappingWrapper(self.env)
        self._max_episode_steps = int(params["max_time_episode"])
        self.action_space = ActionSpace(
            self.env.action_space,
            self.env.action_space.low,
            self.env.action_space.high,
            self.env.action_space.shape,
        )

    def reset(self, ped_task_i=0):
        self.env.set_ped_task_i(ped_task_i)
        obs, _ = self.env.reset()
        return obs

    def step(self, action):
        return self.env.step(action)

    def tasks(self):
        return self.env.get_ped_tasks()

    def curr_task(self):
        return self.env.get_ped_task_i()

    def is_success(self):
        return self.env.get_is_success()
