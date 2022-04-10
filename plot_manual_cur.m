close all;
clear all;

smooth = 1;
cutoff = 70;

runs = {'run-tensorboard_mar7_no_agc-tag-eval_avg_episode_progress.csv', 'run-tensorboard_mar8_agc-tag-eval_avg_episode_progress.csv'};
plot_runs(runs, {'random', 'ACG'}, "Avg Episode Progress", smooth, cutoff)

runs = {'run-tensorboard_mar7_no_agc-tag-eval_episode_reward.csv', 'run-tensorboard_mar8_agc-tag-eval_episode_reward.csv'};
plot_runs(runs, {'random', 'ACG'}, "Avg Episode Reward", smooth, cutoff)