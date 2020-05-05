"""Minimal script to solve and render an environment.
"""

import numpy as np
import time

import assistance_games.envs as envs
from assistance_games.parser import read_pomdp
from assistance_games.solver import pbvi, exact_vi, deep_rl_solve, get_venv
from assistance_games.utils import get_asset


def run_environment(env, policy=None, n_episodes=5, dt=0.1, max_steps=100, render=True):
    def render_fn():
        if render:
            env.render()
            time.sleep(dt)

    for _ in range(n_episodes):
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
            render_fn()
            step += 1

    return None


def run(env_name, algo_name, **kwargs):
    env_fns = {
        'tiger' : (lambda : read_pomdp(get_asset('pomdps/tiger.pomdp'))),
        'fourthree' : envs.FourThreeMaze,
        'redblue' : envs.RedBlueAssistanceProblem,
        'wardrobe' : envs.WardrobeAssistanceProblem,

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
        # Necessary for using LSTMs
        env = get_venv(env)
    else:
        env = env_fns[env_name](use_belief_space=True)

    policy = algo(env)
    run_environment(env, policy, dt=0.5, n_episodes=100)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='redblue')
    parser.add_argument('--algo_name', type=str, default='pbvi')
    args = parser.parse_args()

    run(args.env_name, args.algo_name)


if __name__ == '__main__':
    main()
