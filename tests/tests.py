"""Minimal smoke tests.
"""

from functools import partial
import numpy as np

from assistance_games.core.core2 import ReducedAssistancePOMDP, ReducedFullyObservableAssistancePOMDPWithMatrices
import assistance_games.envs as envs
from assistance_games.solver import pbvi, exact_vi, deep_rl_solve, get_venv
from assistance_games.parser import read_pomdp

deep_rl_100k = partial(deep_rl_solve, total_timesteps=100000)


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
    lower_reward = 1.7
    upper_reward = 2.1
    policy = pbvi(env)
    reward = eval_policy(policy, env, n_eval_episodes=100)
    assert lower_reward < reward < upper_reward


def test_redblue_assistance_problem_reward():
    env = ReducedFullyObservableAssistancePOMDPWithMatrices(envs.RedBlue2())
    target_reward = 2.0
    policy = pbvi(env)
    reward = eval_policy(policy, env)
    assert abs(reward - target_reward) < 0.1


def test_similar_rewards_redblue():
    solvers = (pbvi, exact_vi, deep_rl_100k)

    returns = []
    for solver in solvers:
        env = envs.RedBlue2()
        if solver == deep_rl_100k:
            env = get_venv(ReducedAssistancePOMDP(env))
        else:
            env = ReducedFullyObservableAssistancePOMDPWithMatrices(env)

        policy = solver(env)
        ret = eval_policy(policy, env, n_eval_episodes=100)
        returns.append(ret)

    assert max(returns) - min(returns) < 0.1

if __name__ == '__main__':
    test_fourthree_reward()
    test_redblue_assistance_problem_reward()
    test_similar_rewards_redblue()
