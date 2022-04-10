close all;
clear all;

smooth = 20;
cutoff = 70;

runs = {'run-tensorboard_feb18_no_cir_terminal_200-tag-eval_episode_success_rate.csv', 'run-tensorboard_feb20_cir_b_terminal_10-tag-eval_episode_success_rate.csv'};
plot_runs(runs, {'scratch', 'curriculum'}, "Avg Episode Success Rate", smooth, cutoff)

runs = {'run-tensorboard_feb18_no_cir_terminal_200-tag-eval_episode_reward.csv', 'run-tensorboard_feb20_cir_b_terminal_10-tag-eval_episode_reward.csv'};
plot_runs(runs, {'scratch', 'curriculum'}, "Avg Episode Reward", smooth, cutoff)