"""Minimal script to solve and render an environment.
"""

import functools
import numpy as np
import time

import assistance_games.envs as envs
from assistance_games.parser import read_pomdp
from assistance_games.solver import pbvi, exact_vi, deep_rl_solve
from assistance_games.utils import get_asset


def run_environment(env, policy=None, n_episodes=5, dt=0.1, max_steps=100, render=True):
    def render_fn():
        if render:
            env.render()
            time.sleep(dt)

    for ep in range(n_episodes):
        print('starting ep {}'.format(ep))
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
        'fourthree' : (lambda : envs.FourThreeMaze()),
        'redblue' : (lambda : envs.RedBlueAssistanceProblem()),
        'wardrobe' : (lambda : envs.WardrobeAssistanceProblem()),
        'cakepizza': (lambda : envs.CakePizzaGraphProblem()),
        'cakepizzatimedep': (lambda: envs.CakePizzaTimeDependentProblem()),
        'cakepizzagrid': (lambda: envs.CakePizzaGridProblem())
    }
    algos = {
        'exact' : exact_vi,
        'pbvi' : functools.partial(pbvi, max_iter=4),
        'deeprl' : deep_rl_solve,
        'random' : lambda _ : None,
    }

    env = env_fns[env_name]()
    algo = algos[algo_name]

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
