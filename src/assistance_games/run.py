"""Minimal script to solve and render an environment.
"""

import numpy as np
import time
from pathlib import Path

from assistance_games.parser import read_pomdp
from assistance_games.solver import pbvi, exact_vi, deep_rl_solve, get_venv
from assistance_games.utils import get_asset
from stable_baselines.bench import Monitor

from assistance_games.envs.meal_choice_graph import MealChoiceTimeDependentProblem
from assistance_games.envs.meal_drink_grid import MealDrinkGridProblem
from assistance_games.envs.meal_drink_h_acts import MealDrinkGridHumanMovesProblem
from assistance_games.envs.toy_envs import FourThreeMaze, RedBlueAssistanceProblem, WardrobeAssistanceProblem


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


def run(env_name, algo_name, seed, **kwargs):
    env_fns = {
        'tiger' : (lambda : read_pomdp(get_asset('pomdps/tiger.pomdp'))),
        'fourthree' : FourThreeMaze,
        'redblue' : RedBlueAssistanceProblem,
        'wardrobe' : WardrobeAssistanceProblem,
        'mealgraph': MealChoiceTimeDependentProblem,
        'mealdrink': MealDrinkGridProblem,
        'mealdrinkhmoves': MealDrinkGridHumanMovesProblem
    }
    algos = {
        'exact' : exact_vi,
        'pbvi' : pbvi,
        'deeprl' : deep_rl_solve,
        'random' : lambda _ : None,
    }

    algo = algos[algo_name]

    if algo_name == 'deeprl':
        # We want deeprl to learn the optimal policy without
        # being helped on tracking beliefs
        env = env_fns[env_name](use_belief_space=False)
        # Set up logging
        log_dir = './logs/'
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        env = Monitor(env, log_dir)
        # Necessary for using LSTMs
        env = get_venv(env, n_envs=1)
    else:
        env = env_fns[env_name](use_belief_space=True)

    for seed in range(2):

        print('\n seed {}'.format(seed))
        np.random.seed(seed)
        policy = algo(env, seed=seed) if algo_name == 'deeprl' else algo(env)
        run_environment(env, policy, dt=0.5, n_episodes=5)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--env_name', type=str, default='redblue')
    parser.add_argument('-a', '--algo_name', type=str, default='pbvi')
    parser.add_argument('-s', '--seed', type=int, default=0)
    args = parser.parse_args()

    run(args.env_name, args.algo_name, args.seed)


if __name__ == '__main__':
    main()
