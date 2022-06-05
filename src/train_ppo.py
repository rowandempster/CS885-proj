#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# modified from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail

from collections import deque
import numpy as np
import paddle
import gym
import time
# from mujoco_model import MujocoModel
# from parl.ex import MujocoAgent
from ppo_agent import PPOAgent
from storage import RolloutStorage, RolloutStorageOG
# from parl.algorithms import PPO
from paddle_ppo import PaddlePPO
from env_config import EnvConfig
from env_utils import ParallelEnv, LocalEnv
from ppo_model import PPOModel
from parl.env.mujoco_wrappers import wrap_rms, get_ob_rms
from parl.utils import summary, tensorboard
from paddle_base import PaddleModel
import argparse

LR = 3e-4
GAMMA = 0.99
EPS = 1e-5  # Adam optimizer epsilon (default: 1e-5)
GAE_LAMBDA = 0.95  # Lambda parameter for calculating N-step advantage
ENTROPY_COEF = 0.00  # Entropy coefficient (ie. c_2 in the paper)
VALUE_LOSS_COEF = 0.5  # Value loss coefficient (ie. c_1 in the paper)
MAX_GRAD_NROM = 0.5  # Max gradient norm for gradient clipping
NUM_STEPS = 2048  # data collecting time steps (ie. T in the paper)
PPO_EPOCH = 10  # number of epochs for updating using each T data (ie K in the paper)
CLIP_PARAM = 0.2  # epsilon in clipping loss (ie. clip(r_t, 1 - epsilon, 1 + epsilon))
BATCH_SIZE = 32

EXP_NAME = "mar22_random_vehicle_ppo"
TENSORBOARD_DIR = f"tensorboard_{EXP_NAME}"
EVAL_EPISODES = 10

# Logging Params
LOG_INTERVAL = 1


def evaluate(agent, ob_rms):
    eval_env = gym.make(args.env)
    eval_env.seed(args.seed + 1)
    eval_env = wrap_rms(eval_env, GAMMA, test=True, ob_rms=ob_rms)
    eval_episode_rewards = []
    obs = eval_env.reset()

    while len(eval_episode_rewards) < 10:
        action = agent.predict(obs)

        # Observe reward and next obs
        obs, _, done, info = eval_env.step(action)
        # get validation rewards from info['episode']['r']
        if done:
            eval_episode_rewards.append(info['episode']['r'])

    eval_env.close()

def run_random_evaluate_episodes(agent, env, eval_episodes):
    avg_reward = 0.0
    avg_steps = 0.0
    avg_success = 0.0
    progress_per_episode = []
    max_speed_per_episode = []
    avg_speed = 0.0
    avg_collision = 0.0
    for k in range(eval_episodes):
        obs = env.reset()
        done = False
        steps = 0
        state_info = None
        max_speed = 0
        while not done and steps < env._max_episode_steps:
            steps += 1
            action = agent.predict(obs)
            obs, reward, done, state_info = env.step(action)
            avg_reward += reward
            avg_speed += np.linalg.norm(state_info["velocity_t"])
            if np.linalg.norm(state_info["velocity_t"]) > max_speed:
                max_speed = np.linalg.norm(state_info["velocity_t"])
        if env.env.isSuccess:
            avg_success += 1.0
        if env.env.isCollided:
            avg_collision += 1.0
        progress_per_episode.append(state_info["progress"])
        max_speed_per_episode.append(max_speed)
        avg_steps += steps
    avg_speed /= avg_steps
    avg_reward /= eval_episodes
    avg_steps /= eval_episodes
    avg_success /= eval_episodes
    avg_collision /= eval_episodes
    progress_per_episode = np.array(progress_per_episode)
    max_speed_per_episode = np.array(max_speed_per_episode)
    return {
        "avg_reward": avg_reward,
        "avg_steps": avg_steps,
        "avg_success": avg_success,
        "max_progress": progress_per_episode.max(),
        "avg_progress": np.average(progress_per_episode),
        "avg_speed": avg_speed,
        "max_speed": max_speed_per_episode.max(),
        "average_max_speed": np.average(max_speed_per_episode),
        "avg_collision": avg_collision,
    }

def main():
    # paddle.seed(args.seed)

    # env for eval
    eval_env_params = EnvConfig["eval_env_params"]
    eval_env_params["host"] = "watod_rowan_carla_server_1"
    eval_env = LocalEnv(args.env, eval_env_params)
    eval_env_params["host"] = "watod_rowan_carla_server_2"
    real_eval_env = LocalEnv(args.env, eval_env_params)

    tensorboard.logger.set_dir(TENSORBOARD_DIR)

    obs_dim = eval_env.obs_dim
    action_dim = eval_env.action_dim
    model = PPOModel(obs_dim,
                        action_dim)

    algorithm = PaddlePPO(model, CLIP_PARAM, VALUE_LOSS_COEF, ENTROPY_COEF, LR, EPS,
                    MAX_GRAD_NROM)

    agent = PPOAgent(algorithm)

    rollouts = RolloutStorage(NUM_STEPS, obs_dim,
                              action_dim)

    rollouts = RolloutStorageOG(NUM_STEPS, obs_dim,
                              action_dim)


    obs = eval_env.reset()
    rollouts.obs[0] = np.copy(obs)


    episode_rewards = deque(maxlen=10)
    num_updates = int(args.train_total_steps) // NUM_STEPS
    for j in range(num_updates):
        for step in range(NUM_STEPS):
            # Sample actions
            value, action, action_log_prob = agent.sample(rollouts.obs[step])

            # Obser reward and next obs
            obs, reward, done, info = eval_env.step(action[0])
            # get training rewards from info['episode']['r']

            # If done then clean the history of observations.
            # TODO: Revert 1.0 to 0.0 if doesn't work
            masks = paddle.to_tensor(
                [[0.0]] if done else [[1.0]], dtype='float32')
            rollouts.append(obs, action, action_log_prob, value, reward, masks)
            if done:
                episode_rewards.append(info['progress'])
                obs = eval_env.reset()
                rollouts.obs[step+1] = np.copy(obs)

        next_value = agent.value(rollouts.obs[-1])

        value_loss, action_loss, dist_entropy = agent.learn(
            next_value, GAMMA, GAE_LAMBDA, PPO_EPOCH, BATCH_SIZE, rollouts)

        rollouts.after_update()

        if j % LOG_INTERVAL == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * NUM_STEPS
            print(
                "Updates {}, num timesteps {},\n Last {} training episodes: mean/median reward {}/{}, min/max reward {}/{}\n"
                .format(j, total_num_steps, len(episode_rewards),
                        np.mean(episode_rewards), np.median(episode_rewards),
                        np.min(episode_rewards), np.max(episode_rewards),
                        dist_entropy, value_loss, action_loss))

        if (args.test_every_steps is not None and len(episode_rewards) > 1
                and j % args.test_every_steps == 0):
            total_steps = (j + 1) * NUM_STEPS
            print("running eval")
            eval_runtime = time.time()
            metrics = run_random_evaluate_episodes(
                agent, real_eval_env, EVAL_EPISODES
            )
            eval_runtime = time.time() - eval_runtime
            tensorboard.add_scalar(
                "eval/runtime", eval_runtime, total_steps
            )
            tensorboard.add_scalar(
                "eval/episode_reward", metrics["avg_reward"], total_steps
            )
            tensorboard.add_scalar(
                "eval/episode_steps", metrics["avg_steps"], total_steps
            )
            tensorboard.add_scalar(
                "eval/episode_success_rate", metrics["avg_success"], total_steps
            )
            tensorboard.add_scalar(
                "eval/max_episode_progress", metrics["max_progress"], total_steps
            )
            tensorboard.add_scalar(
                "eval/avg_episode_progress", metrics["avg_progress"], total_steps
            )
            tensorboard.add_scalar(
                "eval/max_speed", metrics["max_speed"], total_steps
            )
            tensorboard.add_scalar(
                "eval/avg_speed", metrics["avg_speed"], total_steps
            )
            tensorboard.add_scalar(
                "eval/episode_collision_rate",
                metrics["avg_collision"],
                total_steps,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xparl_addr",
        default="localhost:8080",
        help="xparl address for parallel training",
    )
    parser.add_argument("--env", default="carla-v0")
    parser.add_argument(
        "--train_total_steps",
        default=5e8,
        type=int,
        help="max time steps to run environment",
    )
    parser.add_argument(
        "--test_every_steps",
        default=10,
        type=int,
        help="the step interval between two consecutive evaluations",
    )
    args = parser.parse_args()

    main()