close all;
clear all;

smooth = 2;
cutoff = 70;

runs = {'run-tensorboard_ap03_mpc_sigmoid-tag-eval_episode_success_rate.csv', 'run-tensorboard_mar15_random_vehicle_alpha_schedule-tag-eval_episode_success_rate.csv'};
plot_runs(runs, {'w/ MPC', 'w/out MPC'}, "Avg Episode Success Rate", smooth, cutoff)
% return;
runs = {'run-tensorboard_ap03_mpc_sigmoid-tag-eval_avg_speed.csv', 'run-tensorboard_mar15_random_vehicle_alpha_schedule-tag-eval_avg_speed.csv'};
plot_runs(runs, {'w/ MPC', 'w/out MPC'}, "Avg Episode Speed", smooth, cutoff)