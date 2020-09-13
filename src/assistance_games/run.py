"""Minimal script to solve and render an environment.
"""

from functools import partial
from pathlib import Path
import os
import time

import numpy as np

from assistance_games.parser import read_pomdp
from assistance_games.solver import pbvi, exact_vi, deep_rl_solve, get_venv
from assistance_games.utils import get_asset

import assistance_games.envs as envs
from assistance_games.core.core2 import ReducedAssistancePOMDP, ReducedAssistancePOMDPWithMatrices, ReducedFullyObservableAssistancePOMDPWithMatrices

def run_environment(env, policy=None, num_episodes=10, dt=0.01, max_steps=100, render=True):
    if num_episodes == -1:
        num_episodes = int(1e6)

    def render_fn(prev_action=None):
        if render:
            env.render(mode='human', prev_action=prev_action)
            time.sleep(dt)

    rewards = []
    for ep in range(num_episodes):
        print('\n starting ep {}'.format(ep))
        total_reward = 0
        ob = env.reset()
        render_fn()

        state = None
        done = False
        step = 0
        while not done and step < max_steps:
            if policy is None:
                ac = env.action_space.sample()
            else:
                ac, state = policy.predict(ob, state)
            ob, re, done, _ = env.step(ac)
            print('r = {}'.format(re))
            total_reward += re
            render_fn(ac)
            step += 1
        rewards.append(total_reward)

    print('Undiscounted rewards: {}\nAverage undiscounted reward: {}'.format(rewards, sum(rewards) / len(rewards)))
    return None


def get_hardcoded_policy(env, env_name, *args, **kwargs):
    if env_name == 'pie':
        return envs.get_pie_hardcoded_robot_policy(env, *args, **kwargs)
    if env_name == 'pie_small':
        return envs.get_small_pie_hardcoded_robot_policy(env, *args, **kwargs)
    if env_name == 'mealgraph2':
        return envs.get_meal_choice_hardcoded_robot_policy(env, *args, **kwargs)
    else:
        raise ValueError("No hardcoded robot policy for this environment.")

def run(
    env_name,
    algo_name,
    seed=0,
    logging=True,
    output_folder='',
    render=True,
    num_episodes=10,
    **kwargs,
):
    if logging is not None:
        log_dir_base = './logs'
        if not output_folder:
            output_folder = env_name
        log_dir = os.path.join(log_dir_base, output_folder, f'seed{seed}')
    else:
        log_dir_base = None
        log_dir = None

    def make_env2_fn(cls):
        def helper(*args, **kwargs):
            apomdp = cls()
            if algo_name not in ('exact', 'pbvi'):
                return ReducedAssistancePOMDP(apomdp)
            elif apomdp.fully_observable:
                return ReducedFullyObservableAssistancePOMDPWithMatrices(apomdp)
            else:
                return ReducedAssistancePOMDPWithMatrices(apomdp)
        return helper

    env_fns = {
        # 'tiger' : (lambda *args, **kwargs: read_pomdp(get_asset('pomdps/tiger.pomdp'))),
        # 'fourthree' : envs.FourThreeMaze,
        'redblue' : envs.RedBlueAssistanceProblem,
        'wardrobe' : envs.WardrobeAssistanceProblem,
        'mealgraph': envs.MealChoiceTimeDependentProblem,
        'mealdrink': envs.MealDrinkGridProblem,
        'mealdrinkhmoves': envs.MealDrinkGridHumanMovesProblem,
        'mealperfectquery' : envs.MealDrinkGridPerfectQueryProblem,
        'pie_mdp' : envs.PieMDPAssistanceProblem,
        'pie' : envs.PieGridworldAssistanceProblem,
        'pie_small' : envs.SmallPieGridworldAssistanceProblem,
        'redblue2' : make_env2_fn(envs.RedBlue2),
        'wardrobe2' : make_env2_fn(envs.Wardrobe2),
        'mealgraph2' : make_env2_fn(envs.MealChoice2),
        'pie_small2' : make_env2_fn(envs.SmallPieGridworld2),
    }
    algos = {
        'exact' : exact_vi,
        'pbvi' : pbvi,
        'deeprl' : partial(deep_rl_solve, log_dir=log_dir_base),
        'random' : lambda *args, **kwargs : None,
        'hardcoded' : partial(get_hardcoded_policy, env_name=env_name),
    }

    algo = algos[algo_name]

    if algo_name == 'deeprl':
        # We want deeprl to learn the optimal policy without
        # being helped on tracking beliefs
        env = env_fns[env_name](use_belief_space=False)
        # Set up logging
        if log_dir is not None:
            # This import can take 10+ seconds, so only do it
            # if necessary
            from stable_baselines.bench import Monitor
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            env = Monitor(env, log_dir)
        # Necessary for using LSTMs
        env = get_venv(env, n_envs=1)
    else:
        env = env_fns[env_name](use_belief_space=True)

    print('\n Running algorithm {} with seed {}'.format(algo_name, seed))
    np.random.seed(seed)
    policy = algo(env, seed=seed, **kwargs)
    run_environment(env, policy, dt=0.5, num_episodes=num_episodes, render=render)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--env_name', type=str, default='redblue')
    parser.add_argument('-a', '--algo_name', type=str, default='pbvi')
    parser.add_argument('-o', '--output_folder', type=str, default='')
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-n', '--total_timesteps', type=int, default=int(1e6))
    parser.add_argument('-m', '--num_runs', type=int, default=1)
    parser.add_argument('-p', '--num_episodes', type=int, default=10)
    parser.add_argument('-r', '--render', default=True, action='store_true')
    parser.add_argument('-nr', '--no_render', dest='render', action='store_false')
    parser.add_argument('-l', '--logging', default=True, action='store_true')
    parser.add_argument('-nl', '--no_logging', dest='logging', action='store_false')
    args = parser.parse_args()

    for run_id in range(args.num_runs):
        seed = args.seed + run_id
        run(
            env_name=args.env_name,
            algo_name=args.algo_name,
            seed=seed,
            logging=args.logging,
            output_folder=args.output_folder,
            total_timesteps=args.total_timesteps,
            render=args.render,
            num_episodes=args.num_episodes,
        )


if __name__ == '__main__':
    main()
