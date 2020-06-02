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

def run_environment(env, policy=None, n_episodes=None, dt=0.01, max_steps=100, render=True):
    if n_episodes is None:
        n_episodes = int(1e30)

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
        total_re = 0
        while not done and step < max_steps:
            if policy is None:
                ac = env.action_space.sample()
            else:
                ac, state = policy.predict(ob, state)
            old_ob = ob
            ob, re, done, _ = env.step(ac)
            print('r = {}'.format(re))
            total_re += re
            step += 1
            print(step)
            render_fn()
        print(f'total_re: {total_re}')
    return None


def get_hardcoded_policy(env, *args, **kwargs):
    if isinstance(env, envs.PieGridworldAssistanceProblem):
        return envs.get_pie_hardcoded_robot_policy(env, *args, **kwargs)
    if isinstance(env, envs.SmallPieGridworldAssistanceProblem):
        return envs.get_smallpie_hardcoded_robot_policy(env, *args, **kwargs)
    if isinstance(env, envs.MiniPieGridworldAssistanceProblem):
        return envs.get_minipie_hardcoded_robot_policy(env, *args, **kwargs)
    else:
        raise Error("No hardcoded robot policy for this environment.")

def run(
    env_name,
    algo_name,
    seed=0,
    logging=True,
    output_folder='',
    render=True,
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
        'fourthree' : envs.FourThreeMaze,
        'redblue' : envs.RedBlueAssistanceProblem,
        'wardrobe' : envs.WardrobeAssistanceProblem,
        'mealgraph': envs.MealChoiceTimeDependentProblem,
        'mealdrink': envs.MealDrinkGridProblem,
        'mealdrinkhmoves': envs.MealDrinkGridHumanMovesProblem,
        'mealperfectquery' : envs.MealDrinkGridPerfectQueryProblem,
        'pie_mdp' : envs.PieMDPAssistanceProblem,
        'pie' : envs.PieGridworldAssistanceProblem,
        'small_pie' : envs.SmallPieGridworldAssistanceProblem,
        'mini_pie' : envs.MiniPieGridworldAssistanceProblem,
    }
    algos = {
        'exact' : exact_vi,
        'pbvi' : pbvi,
        'deeprl' : partial(deep_rl_solve, log_dir=log_dir_base),
        'lstm-ppo' : partial(deep_rl_solve, log_dir=log_dir_base),
        'ppo' : partial(deep_rl_solve, log_dir=log_dir_base, use_lstm=False),
        'random' : lambda *args, **kwargs : None,
        'hardcoded' : get_hardcoded_policy,
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
    policy = algo(env, seed=seed, **kwargs)
    run_environment(env, policy, dt=0.5, n_episodes=None, render=render)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--env_name', type=str, default='redblue')
    parser.add_argument('-a', '--algo_name', type=str, default='pbvi')
    parser.add_argument('-o', '--output_folder', type=str, default='')
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-n', '--total_timesteps', type=int, default=int(1e6))
    parser.add_argument('-r', '--render', default=True, action='store_true')
    parser.add_argument('-nr', '--no_render', dest='render', action='store_false')
    parser.add_argument('-l', '--logging', default=True, action='store_true')
    parser.add_argument('-nl', '--no_logging', dest='logging', action='store_false')
    args = parser.parse_args()

    run(
        env_name=args.env_name,
        algo_name=args.algo_name,
        seed=args.seed,
        logging=args.logging,
        output_folder=args.output_folder,
        total_timesteps=args.total_timesteps,
        render=args.render,
    )


if __name__ == '__main__':
    main()
