#!/bin/sh
env="BVRSim"
scenario="default"
num_agents_each_team=2
algo="mqmix"
exp="debug"
seed_max=1

echo "env is ${env}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"

for seed in $(seq ${seed_max}); do
    echo "seed is ${seed}:"
    python ./train_bvr.py \
        --env_name ${env} \
        --algorithm_name ${algo} \
        --experiment_name ${exp} \
        --scenario_name ${scenario} \
        --num_agents_each_team ${num_agents_each_team} \
        --seed ${seed} \
        --n_rollout_threads 4 \
        --episode_length 400 \
        --batch_size 32 \
        --tau 0.005 \
        --lr 7e-4 \
        --hard_update_interval_episode 100 \
        --num_env_steps 5000000 \
        --use_reward_normalization \
        --use_wandb
    echo "training is done!"
done
