close all;
clear all;

runs = {'run-tensorboard_mar24_random_vehicle_td3-tag-eval_avg_episode_progress.csv',
    'run-tensorboard_mar22_random_vehicle_ppo-tag-eval_avg_episode_progress.csv', 
    'run-tensorboard_mar15_random_vehicle_alpha_schedule-tag-eval_avg_episode_progress.csv'};
plot_runs(runs, {'td3', 'ppo', 'sac'}, "Avg Episode Progress")

runs = {'run-tensorboard_mar24_random_vehicle_td3-tag-eval_episode_reward.csv',
    'run-tensorboard_mar22_random_vehicle_ppo-tag-eval_episode_reward.csv', 
    'run-tensorboard_mar15_random_vehicle_alpha_schedule-tag-eval_episode_reward.csv'};
plot_runs(runs, {'td3', 'ppo', 'sac'}, "Avg Episode Reward")

runs = {'run-tensorboard_mar24_random_vehicle_td3-tag-eval_episode_success_rate.csv',
    'run-tensorboard_mar22_random_vehicle_ppo-tag-eval_episode_success_rate.csv', 
    'run-tensorboard_mar15_random_vehicle_alpha_schedule-tag-eval_episode_success_rate.csv'};
plot_runs(runs, {'td3', 'ppo', 'sac'}, "Avg Episode Success Rate")

runs = {'run-tensorboard_mar24_random_vehicle_td3-tag-eval_episode_collision_rate.csv',
    'run-tensorboard_mar22_random_vehicle_ppo-tag-eval_episode_collision_rate.csv', 
    'run-tensorboard_mar15_random_vehicle_alpha_schedule-tag-eval_episode_collision_rate.csv'};
plot_runs(runs, {'td3', 'ppo', 'sac'}, "Avg Episode Collision Rate")

runs = {'run-tensorboard_mar24_random_vehicle_td3-tag-eval_avg_speed.csv',
    'run-tensorboard_mar22_random_vehicle_ppo-tag-eval_avg_speed.csv', 
    'run-tensorboard_mar15_random_vehicle_alpha_schedule-tag-eval_avg_speed.csv'};
plot_runs(runs, {'td3', 'ppo', 'sac'}, "Avg Episode Speed")