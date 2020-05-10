"""Minimal smoke tests.
"""

from functools import partial
import numpy as np

from assistance_games.core import BeliefRewardModel, TabularRewardModel
import assistance_games.envs as envs
from assistance_games.solver import pbvi, exact_vi, deep_rl_solve, get_venv
from assistance_games.parser import read_pomdp


def eval_policy(
    policy,
    env,
    n_eval_episodes=10,
    use_discount=False,
    max_steps=float('inf'),
):
    if hasattr(env, 'reward_model') and isinstance(env.reward_model, BeliefRewardModel):
        # Belief MDPs get reward based on current belief; we
        # want to evaluate on the true reward instead
        env = set_true_reward_model(env)

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


def set_true_reward_model(env):
    env.reward_model = TabularRewardModel(env, reward_matrix=env.reward_model.reward_matrix)
    return env


def test_fourthree_reward():
    env = envs.FourThreeMaze(horizon=20)
    lower_reward = 1.7
    upper_reward = 2.0
    policy = pbvi(env)
    reward = eval_policy(policy, env, n_eval_episodes=100)
    assert lower_reward < reward < upper_reward


def test_redblue_assistance_problem_reward():
    env = envs.RedBlueAssistanceProblem()
    target_reward = 2.0
    policy = pbvi(env)
    env = set_true_reward_model(env)
    reward = eval_policy(policy, env)
    assert abs(reward - target_reward) < 0.1

def test_similar_rewards_fourthree():
    solvers = (pbvi, deep_rl_solve)

    returns = []

    for solver in solvers:
        env = envs.FourThreeMaze(horizon=20)
        if solver == deep_rl_solve:
            env = get_venv(env)

        policy = solver(env)
        ret = eval_policy(policy, env, n_eval_episodes=100)
        returns.append(ret)

    assert max(returns) - min(returns) < 0.1


def test_similar_rewards_redblue():
    solvers = (pbvi, exact_vi, deep_rl_solve)

    returns = []
    for solver in solvers:
        if solver == deep_rl_solve:
            env = envs.RedBlueAssistanceProblem(use_belief_space=False)
            env = get_venv(env)
        else:
            env = envs.RedBlueAssistanceProblem(use_belief_space=True)

        policy = solver(env)
        ret = eval_policy(policy, env, n_eval_episodes=100)
        returns.append(ret)

    assert max(returns) - min(returns) < 0.1
