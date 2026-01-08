import sys
import os
import numpy as np
from pathlib import Path
import socket
import wandb
import setproctitle
import torch
from offpolicy.config import get_config
from offpolicy.utils.util import get_cent_act_dim, get_dim_from_space
from offpolicy.envs.bvr_sim.env_marlbenchmark import BVRSimEnvMarlBenchmark, MultiAgentEnvWrapper
from offpolicy.envs.env_wrappers import DummyVecEnv, SubprocVecEnv


def get_env_config(all_args):
    env_config = {
        'N_EACH_TEAM': [all_args.num_agents_each_team],
        'MaxEpisodeStep': all_args.episode_length,
        'render': True,
        'dt': 0.4,
        'cpp_env_init_file': "offpolicy/envs/bvr_sim/conf_system/cpp/init/2v2.jsonc",
        'obs_type': 'entity',
        'blue_opponent_type': 'tactical',
    }
    return env_config

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "BVRSim":
                env_config = get_env_config(all_args)
                base_env = BVRSimEnvMarlBenchmark(rank, env_config)
                env = MultiAgentEnvWrapper(base_env)
            else:
                print("Can not support the " + all_args.env_name + " environment.")
                raise NotImplementedError
            env.seed(all_args.seed + abs(rank) * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "BVRSim":
                env_config = get_env_config(all_args)
                base_env = BVRSimEnvMarlBenchmark(rank, env_config)
                env = MultiAgentEnvWrapper(base_env)
            else:
                print("Can not support the " + all_args.env_name + " environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + abs(rank) * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(-1)])
    else:
        return SubprocVecEnv([get_env_fn(-i-1) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--num_agents_each_team', type=int, default=2,
                        help="number of agents in each team")
    parser.add_argument('--scenario_name', type=str, default='default',
                        help="scenario name (not used for BVRSim but kept for compatibility)")
    # parser.add_argument('--dt', type=float, default=0.4,
    #                     help="simulation time step")
    # parser.add_argument('--field_size', type=float, default=100000.0,
    #                     help="battlefield size")
    # parser.add_argument('--cpp_env_init_file', type=str, default=None,
    #                     help="C++ environment init file (if using C++ backend)")
    # parser.add_argument('--obs_type', type=str, default='entity',
    #                     help="observation type: entity, compact, extended")
    # parser.add_argument('--blue_opponent_type', type=str, default='tactical',
    #                     help="blue team opponent type: random, simple, tactical")
    parser.add_argument('--use_same_share_obs', action='store_false',
                        default=True, help="Whether agents share same observation")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    run_dir = Path(os.path.dirname(os.path.abspath(__file__)) + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name

    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                         str(all_args.experiment_name) +
                         "_seed" + str(all_args.seed),
                         group=all_args.scenario_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[
                                 1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + str(
        all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    env = make_train_env(all_args)
    num_agents = all_args.num_agents_each_team

    if all_args.share_policy:
        policy_info = {
            'policy_0': {"cent_obs_dim": get_dim_from_space(env.share_observation_space[0]),
                         "cent_act_dim": get_cent_act_dim(env.action_space),
                         "obs_space": env.observation_space[0],
                         "share_obs_space": env.share_observation_space[0],
                         "act_space": env.action_space[0]}
        }

        def policy_mapping_fn(id): return 'policy_0'
    else:
        policy_info = {
            'policy_' + str(agent_id): {"cent_obs_dim": get_dim_from_space(env.share_observation_space[agent_id]),
                                        "cent_act_dim": get_cent_act_dim(env.action_space),
                                        "obs_space": env.observation_space[agent_id],
                                        "share_obs_space": env.share_observation_space[agent_id],
                                        "act_space": env.action_space[agent_id]}
            for agent_id in range(num_agents)
        }

        def policy_mapping_fn(agent_id): return 'policy_' + str(agent_id)

    if all_args.algorithm_name in ["rmatd3", "rmaddpg", "rmasac", "qmix", "vdn"]:
        from offpolicy.runner.rnn.bvr_runner import BVRRunner as Runner
        assert all_args.n_rollout_threads == 1, (
            "only support 1 env in recurrent version.")
        eval_env = env
    elif all_args.algorithm_name in ["matd3", "maddpg", "masac", "mqmix", "mvdn"]:
        from offpolicy.runner.mlp.bvr_runner import BVRRunner as Runner
        eval_env = make_eval_env(all_args)
    else:
        raise NotImplementedError

    config = {"args": all_args,
              "policy_info": policy_info,
              "policy_mapping_fn": policy_mapping_fn,
              "env": env,
              "eval_env": eval_env,
              "num_agents": num_agents,
              "device": device,
              "use_same_share_obs": all_args.use_same_share_obs,
              "run_dir": run_dir
              }

    total_num_steps = 0
    runner = Runner(config=config)
    while total_num_steps < all_args.num_env_steps:
        total_num_steps = runner.run()

    env.close()
    if all_args.use_eval and (eval_env is not env):
        eval_env.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(
            str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
