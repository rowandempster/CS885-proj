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

import gym
import argparse
import numpy as np
from parl.utils import logger, summary, ReplayMemory, tensorboard
from parl.env.continuous_wrappers import ActionMappingWrapper
from paddle_base import PaddleModel, PaddleSAC, PaddleAgent, PaddlePPO
from td3_model import TD3Model
from td3_agent import TD3Agent
import time
from env_config import EnvConfig
from env_utils import ParallelEnv, LocalEnv
from parl.algorithms import TD3

WARMUP_STEPS = 1e4
MEMORY_SIZE = int(1e6)
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.005
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
EXPL_NOISE = 0.1

EXP_NAME = "mar24_random_vehicle_td3"
TENSORBOARD_DIR = f"tensorboard_{EXP_NAME}"
EVAL_EPISODES = 10


# Run episode for training
def run_train_episode(agent, env, rpm):
    action_dim = env.action_dim
    obs = env.reset()
    done = False
    episode_reward = 0
    episode_steps = 0
    while not done:
        episode_steps += 1
        # Select action randomly or according to policy
        if rpm.size() < WARMUP_STEPS:
            action = np.random.uniform(-1, 1, size=action_dim)
        else:
            action = agent.sample(obs)

        # Perform action
        next_obs, reward, done, _ = env.step(action)
        terminal = float(done) if episode_steps < env._max_episode_steps else 0

        # Store data in replay memory
        rpm.append(obs, action, reward, next_obs, terminal)

        obs = next_obs
        episode_reward += reward

        # Train agent after collecting sufficient data
        if rpm.size() >= WARMUP_STEPS:
            batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal = rpm.sample_batch(
                BATCH_SIZE)
            agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
                        batch_terminal)

    return episode_reward, episode_steps

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
    logger.info("------------------- TD3 ---------------------")
    logger.info('Env: {}, Seed: {}'.format(args.env, args.seed))
    logger.info("---------------------------------------------")
    logger.set_dir('./{}_{}'.format(args.env, args.seed))

    eval_env_params = EnvConfig["eval_env_params"]
    eval_env_params["host"] = "watod_rowan_carla_server_1"
    eval_env = LocalEnv(args.env, eval_env_params)
    eval_env_params["host"] = "watod_rowan_carla_server_2"
    real_eval_env = LocalEnv(args.env, eval_env_params)

    tensorboard.logger.set_dir(TENSORBOARD_DIR)

    obs_dim = eval_env.obs_dim
    action_dim = eval_env.action_dim

    # Initialize model, algorithm, agent, replay_memory
    model = TD3Model(obs_dim, action_dim)
    algorithm = TD3(
        model,
        gamma=GAMMA,
        tau=TAU,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR,
        policy_freq=args.policy_freq)
    agent = TD3Agent(algorithm, act_dim=action_dim, expl_noise=EXPL_NOISE)
    rpm = ReplayMemory(
        max_size=MEMORY_SIZE, obs_dim=obs_dim, act_dim=action_dim)

    total_steps = 0
    test_flag = 0
    while total_steps < args.train_total_steps:
        # Train episode
        episode_reward, episode_steps = run_train_episode(agent, eval_env, rpm)
        total_steps += episode_steps

        summary.add_scalar('train/episode_reward', episode_reward, total_steps)
        logger.info('Total Steps: {} Reward: {}'.format(
            total_steps, episode_reward))

        # Evaluate episode
        if (total_steps + 1) // args.test_every_steps >= test_flag:
            while (total_steps + 1) // args.test_every_steps >= test_flag:
                test_flag += 1
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
        "--env", default="carla-v0", help='Mujoco gym environment name')
    parser.add_argument("--seed", default=0, type=int, help='Sets Gym seed')
    parser.add_argument(
        "--train_total_steps",
        default=int(3e10),
        type=int,
        help='Max time steps to run environment')
    parser.add_argument(
        '--test_every_steps',
        type=int,
        default=int(1e4),
        help='The step interval between two consecutive evaluations')
    parser.add_argument(
        '--policy_freq',
        type=int,
        default=2,
        help='Frequency of delayed policy updates')
    args = parser.parse_args()

    main()