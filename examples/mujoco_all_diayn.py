"""Script for launching DIAYN experiments.

Usage:
    python mujoco_all_diayn.py --env=point --log_dir=/dev/null
"""
import sys
sys.path.append('../Learning-is-a-MUST')
sys.path.append('../Learning-is-a-MUST/mape')
from swarm.utils import make_multiagent_env
from sac.misc.utils import PseudoSingleAgentEnv


from rllab.envs.env_spec import EnvSpec
from rllab.misc.instrument import VariantGenerator
from rllab.envs.normalized_env import normalize
from rllab import spaces

from sac.algos import DIAYN
from sac.envs.gym_env import GymEnv
from sac.misc.instrument import run_sac_experiment
from sac.misc.utils import timestamp
from sac.policies.gmm import GMMPolicy
from sac.replay_buffers import SimpleReplayBuffer
from sac.value_functions import NNQFunction, NNVFunction, NNDiscriminatorFunction
import sys
import argparse
import numpy as np
import os


# what does number of skills represent here? Guess: number of different skills to be learnt
# rest of the params are the learning rate, discount factor, layer size in the nn, batch size
# not sure about what is epoch length(May be it is episode length)
# scale entropy seems to be the temperature term
# learn p_z specifies whether to learn the distribution of the latent factor.
# not sure what snapshot_mode means here.
# we can add/remove envs from here. Also, change number of epochs.

SHARED_PARAMS = {
    'seed': [1],
    'lr': 3E-4,
    'discount': 0.99,
    'tau': 0.01,
    'K': 4,
    'layer_size': 300,
    'batch_size': 128,
    'max_pool_size': 1E6,
    'n_train_repeat': 1,
    'epoch_length': 1000,
    'snapshot_mode': 'gap',
    'snapshot_gap': 10,
    'sync_pkl': True,
    'num_skills': 50,
    'scale_entropy': 0.1,
    'include_actions': False,
    'learn_p_z': False,
    'add_p_z': True,
}

TAG_KEYS = ['seed']

# specifies env name, path length(in timesteps?), number of epochs.
ENV_PARAMS = {
    'swimmer': {  # 2 DoF
        'prefix': 'swimmer',
        'env_name': 'Swimmer-v1',
        'max_path_length': 1000,
        'n_epochs': 10000,
    },
    'hopper': {  # 3 DoF
        'prefix': 'hopper',
        'env_name': 'Hopper-v1',
        'max_path_length': 1000,
        'n_epochs': 10000,
    },
    'half-cheetah': {  # 6 DoF
        'prefix': 'half-cheetah',
        'env_name': 'HalfCheetah-v1',
        'max_path_length': 1000,
        'n_epochs': 10000,
        'max_pool_size': 1E7,
    },
    'walker': {  # 6 DoF
        'prefix': 'walker',
        'env_name': 'Walker2d-v1',
        'max_path_length': 1000,
        'n_epochs': 10000,
    },
    'ant': {  # 8 DoF
        'prefix': 'ant',
        'env_name': 'Ant-v1',
        'max_path_length': 1000,
        'n_epochs': 10000,
    },
    'humanoid': {  # 21 DoF
        'prefix': 'humanoid',
        'env_name': 'Humanoid-v1',
        'max_path_length': 1000,
        'n_epochs': 20000,
    },
    'point': {
        'prefix': 'point',
        'env_name': 'point-rllab',
        'layer_size': 32,
        'max_path_length': 100,
        'n_epochs': 50,
        'scale_reward': 1,
    },
    'inverted-pendulum': {
        'prefix': 'inverted-pendulum',
        'env_name': 'InvertedPendulum-v1',
        'max_path_length': 1000,
        'n_epochs': 1000,
    },
    'inverted-double-pendulum': {
        'prefix': 'inverted-double-pendulum',
        'env_name': 'InvertedDoublePendulum-v1',
        'max_path_length': 1000,
        'n_epochs': 1000,
    },
    'pendulum': {
        'prefix': 'pendulum',
        'env_name': 'Pendulum-v0',
        'max_path_length': 200,
        'layer_size': 32,
        'n_epochs': 1000,
        'num_skills': 5,
    },
    'mountain-car': {
        'prefix': 'mountain-car',
        'env_name': 'MountainCarContinuous-v0',
        'max_path_length': 1000,
        'n_epochs': 1000,
        'add_p_z': False,
    },
    'lunar-lander': {
        'prefix': 'lunar-lander',
        'env_name': 'LunarLanderContinuous-v2',
        'max_path_length': 1000,
        'n_epochs': 1000,
    },
    'bipedal-walker': {
        'prefix': 'bipedal-walker',
        'env_name': 'BipedalWalker-v2',
        'max_path_length': 1600,
        'n_epochs': 1000,
        'scale_entropy': 0.1,
    },
    'simple_obstacle': {
        'prefix': 'simple_obstacle',
        'env_name': 'simple_obstacle',
        'layer_size': 32,
        'max_path_length': 100,
        'n_epochs': 1000,
        'scale_reward': 1,
        'num_skills': 5,
    },
}
DEFAULT_ENV = 'swimmer'
AVAILABLE_ENVS = list(ENV_PARAMS.keys())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',
                        type=str,
                        choices=AVAILABLE_ENVS,
                        default='swimmer')
    parser.add_argument('--exp_name', type=str, default=timestamp())
    parser.add_argument('--mode', type=str, default='local')
    parser.add_argument('--log_dir', type=str, default=None)
    # parser.add_argument('--use_task_reward', action="store_true", default=False, help='use task reward in diyan')

    args = parser.parse_args()

    return args


def get_variants(args):
    env_params = ENV_PARAMS[args.env]
    params = SHARED_PARAMS
    params.update(env_params)
    # use_task_reward = args.use_task_reward
    # seems something related to management of parameters.
    vg = VariantGenerator()
    for key, val in params.items():
        if isinstance(val, list):
            vg.add(key, val)
        else:
            vg.add(key, [val])

    # vg.add("use_task_reward", use_task_reward)

    return vg


def make_env(env_name):
    if env_name == 'humanoid-rllab':
        from rllab.envs.mujoco.humanoid_env import HumanoidEnv
        env = normalize(HumanoidEnv())
    elif env_name == 'swimmer-rllab':
        from rllab.envs.mujoco.swimmer_env import SwimmerEnv
        env = normalize(SwimmerEnv())
    elif env_name == 'point-rllab':
        from rllab.envs.mujoco.point_env import PointEnv
        env = normalize(PointEnv())
    elif env_name == 'simple_obstacle':
        env = normalize(PseudoSingleAgentEnv(make_multiagent_env('simple_obstacle', 3, 0.1, 1, 0, num_steps=100, diff_reward=False, video_format='gif', discrete_action=False)))
    else:
        env = normalize(GymEnv(env_name))
    return env

def run_experiment(variant):
    env = make_env(variant['env_name'])

    obs_space = env.spec.observation_space
    print('obs_space', obs_space)
    # check the observation space should be continuous.
    assert isinstance(obs_space, spaces.Box)

    # not sure why the state and the action spaces have been augumented?
    # Guess: Each skill's probability will vary from 0 to 1. So low and high are set accordingly.

    low = np.hstack([obs_space.low, np.full(variant['num_skills'], 0)])
    high = np.hstack([obs_space.high, np.full(variant['num_skills'], 1)])
    aug_obs_space = spaces.Box(low=low, high=high)

    # seems like a wrapper to specify the augmented state-action space.
    aug_env_spec = EnvSpec(aug_obs_space, env.spec.action_space)

    # create a replay buffer.
    pool = SimpleReplayBuffer(
        env_spec=aug_env_spec,
        max_replay_buffer_size=variant['max_pool_size'],
    )

    base_kwargs = dict(
        min_pool_size=variant['max_path_length'],
        epoch_length=variant['epoch_length'],
        n_epochs=variant['n_epochs'],
        max_path_length=variant['max_path_length'],
        batch_size=variant['batch_size'],
        n_train_repeat=variant['n_train_repeat'],
        eval_render=False,
        eval_n_episodes=1,
        eval_deterministic=True,
    )

    # qf and vf are for the critic part of the model in the Soft Actor Critic algorithm.
    M = variant['layer_size']
    qf = NNQFunction(
        env_spec=aug_env_spec,
        hidden_layer_sizes=[M, M],
    )

    # this is probably the target network.
    vf = NNVFunction(
        env_spec=aug_env_spec,
        hidden_layer_sizes=[M, M],
    )

    # policy is conditioned on both state and the z value.
    # It is a stochastic policy
    policy = GMMPolicy(
        env_spec=aug_env_spec,
        K=variant['K'],
        hidden_layer_sizes=[M, M],
        qf=qf,
        reg=0.001,
    )

    # Discriminator is used to get the probability of z given state
    discriminator = NNDiscriminatorFunction(
        env_spec=env.spec,
        hidden_layer_sizes=[M, M],
        num_skills=variant['num_skills'],
    )

    algorithm = DIAYN(
        base_kwargs=base_kwargs,
        env=env,
        policy=policy,
        discriminator=discriminator,
        pool=pool,
        qf=qf,
        vf=vf,

        lr=variant['lr'],
        scale_entropy=variant['scale_entropy'],
        discount=variant['discount'],
        tau=variant['tau'],
        num_skills=variant['num_skills'],
        save_full_state=False,
        include_actions=variant['include_actions'],
        learn_p_z=variant['learn_p_z'],
        add_p_z=variant['add_p_z'],
        use_task_reward=True,
    )

    algorithm.train()


def launch_experiments(variant_generator):
    variants = variant_generator.variants()

    for i, variant in enumerate(variants):
        tag = '__'.join(['%s_%s' % (key, variant[key]) for key in TAG_KEYS])
        log_dir = os.path.join(args.log_dir, tag)
        print('Launching {} experiments.'.format(len(variants)))

        # essentially run_experiment is being called. run_sac_experiment additionally is just handling
        # so paths for book keeping.
        run_sac_experiment(
            run_experiment,
            mode=args.mode,
            variant=variant,
            exp_prefix=variant['prefix'] + '/' + args.exp_name,
            exp_name=variant['prefix'] + '-' + args.exp_name + '-' + str(i).zfill(2),
            n_parallel=1,  # Increasing this barely effects performance,
            # but breaks learning of hierarchical policy.
            seed=variant['seed'],
            terminate_machine=True,
            log_dir=log_dir,
            snapshot_mode=variant['snapshot_mode'],
            snapshot_gap=variant['snapshot_gap'],
            sync_s3_pkl=variant['sync_pkl'],
        )


if __name__ == '__main__':
    args = parse_args()

    # get the parameters for the given environment.
    variant_generator = get_variants(args)

    # launch experiments now.
    launch_experiments(variant_generator)
