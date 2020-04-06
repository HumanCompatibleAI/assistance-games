"""Minimal smoke tests.
"""

from functools import partial
import numpy as np

import assistance_games.envs as envs
from assistance_games.solver import pbvi, exact_vi, deep_rl_solve
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


def test_four_three_reward():
    env = envs.FourThreeMaze(horizon=20)
    lower_reward = 1.5
    policy = pbvi(env)
    reward = eval_policy(policy, env, n_eval_episodes=100)
    assert reward > lower_reward


def test_two_balls_assistance_problem_reward():
    env = envs.RedBlueAssistanceProblem()
    # Undiscounted and myopic reward
    target_reward = 1.5
    policy = pbvi(env)
    reward = eval_policy(policy, env)
    assert abs(reward - target_reward) < 1e-3


def test_similar_rewards_four_three():
    env = envs.FourThreeMaze(horizon=20)
    solvers = (pbvi, deep_rl_solve)
    _test_similar_rewards(env, solvers)


def test_similar_rewards_balls():
    env = envs.RedBlueAssistanceProblem()
    solvers = (pbvi, pbvi_bs, exact_vi_bs, deep_rl_solve)
    _test_similar_rewards(env, solvers)


def _test_similar_rewards(env, solvers, n_eval_episodes=100, abs_diff=3e-1):
    rewards = []
    for solver in solvers:
        policy = solver(env)
        reward = eval_policy(policy, env, n_eval_episodes=n_eval_episodes)
        rewards.append(reward)
    assert max(rewards) - min(rewards) < abs_diff
