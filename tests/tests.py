"""Minimal smoke tests.
"""

from functools import partial
import numpy as np

import assistance_games.envs as envs
from assistance_games.solver import pbvi, exact_vi, deep_rl_solve, get_venv
from assistance_games.parser import read_pomdp

pbvi_bs = partial(pbvi, use_back_sensor=True)
exact_vi_bs = partial(exact_vi, use_back_sensor=True)


def eval_policy(
    policy,
    env,
    n_eval_episodes=10,
    use_discount=False,
    max_steps=float('inf'),
):
    discount = env.discount if use_discount else 1.0
    rets = []
    for _ in range(n_eval_episodes):
        ob = env.reset()

        state = None
        done = False
        step = 0
        rews = []
        while not done and step < max_steps:
            ac, state = policy.predict(ob, state)
            ob, rew, done, _ = env.step(ac)
            rews.append(rew)
            step += 1
        rets.append(sum(discount**t * rew for t, rew in enumerate(rews)))

    return np.mean(rets)


def test_fourthree_reward():
    env = envs.FourThreeMaze(horizon=20)
    lower_reward = 1.5
    upper_reward = 2.0
    policy = pbvi(env)
    reward = eval_policy(policy, env, n_eval_episodes=100)
    assert lower_reward < reward < upper_reward


def test_redblue_assistance_problem_reward():
    env = envs.RedBlueAssistanceProblem()
    # Undiscounted and myopic reward
    target_reward = 2.0
    policy = pbvi(env)
    reward = eval_policy(policy, env)
    assert abs(reward - target_reward) < 1e-3


def test_similar_rewards_fourthree():
    env_fn = lambda : envs.FourThreeMaze(horizon=20)
    solvers = (pbvi, deep_rl_solve)
    _test_similar_rewards(env_fn, solvers)


def test_similar_rewards_redblue():
    env_fn = lambda : envs.RedBlueAssistanceProblem()
    solvers = (pbvi, exact_vi_bs, deep_rl_solve)
    _test_similar_rewards(env_fn, solvers)


def _test_similar_rewards(env_fn, solvers, abs_diff=3e-1):
    rewards = [_eval_solver(env_fn, solver) for solver in solvers]
    assert max(rewards) - min(rewards) < abs_diff

def _eval_solver(env_fn, solver):
    env = env_fn()
    if solver == deep_rl_solve:
        env.use_belief_space = False
        env = get_venv(env)

    policy = solver(env)
    ret = eval_policy(policy, env, n_eval_episodes=100)
    return ret
