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

from assistance_games.envs import (
    FourThreeMaze,
    MealChoiceTimeDependentProblem,
    MealDrinkGridHumanMovesProblem,
    MealDrinkGridPerfectQueryProblem,
    MealDrinkGridProblem,
    RedBlueAssistanceProblem,
    WardrobeAssistanceProblem,
)

def run_environment(env, policy=None, n_episodes=10, dt=0.01, max_steps=100, render=True):
    def render_fn():
        if render:
            env.render(mode='human')
            time.sleep(dt)

    for ep in range(n_episodes):
        print('\n starting ep {}'.format(ep))
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
            old_ob = ob
            ob, re, done, _ = env.step(ac)
            print('r = {}'.format(re))
            render_fn()
            step += 1
    return None


def run(
    env_name,
    algo_name,
    seed=0,
    logging=True,
    output_folder='',
    **kwargs,
):
    if logging is not None:
        log_dir_base = './logs'
        log_dir = os.path.join(log_dir_base, output_folder, f'seed{seed}')
    else:
        log_dir_base = None
        log_dir = None

    env_fns = {
        'tiger' : (lambda : read_pomdp(get_asset('pomdps/tiger.pomdp'))),
        'fourthree' : FourThreeMaze,
        'redblue' : RedBlueAssistanceProblem,
        'wardrobe' : WardrobeAssistanceProblem,
        'mealgraph': MealChoiceTimeDependentProblem,
        'mealdrink': MealDrinkGridProblem,
        'mealdrinkhmoves': MealDrinkGridHumanMovesProblem,
        'mealperfectquery' : MealDrinkGridPerfectQueryProblem,
    }
    algos = {
        'exact' : exact_vi,
        'pbvi' : pbvi,
        'deeprl' : partial(deep_rl_solve, log_dir=log_dir_base),
        'random' : lambda _ : None,
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

    print('\n seed {}'.format(seed))
    np.random.seed(seed)
    policy = algo(env, seed=seed, **kwargs) if algo_name == 'deeprl' else algo(env)
    run_environment(env, policy, dt=0.5, n_episodes=5)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--env_name', type=str, default='redblue')
    parser.add_argument('-a', '--algo_name', type=str, default='pbvi')
    parser.add_argument('-o', '--output_folder', type=str, default='')
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-n', '--total_timesteps', type=int, default=int(1e6))
    parser.add_argument('-nl', '--no_logging', action='store_true')
    args = parser.parse_args()
    logging = not args.no_logging

    run(
        env_name=args.env_name,
        algo_name=args.algo_name,
        seed=args.seed,
        logging=logging,
        output_folder=args.output_folder,
        total_timesteps=args.total_timesteps,
    )


if __name__ == '__main__':
    main()
