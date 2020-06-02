import numpy as np
from scipy.special import logsumexp


def tabular_random_policy_fn(assistance_game, reward):
    num_states = assistance_game.state_space.n
    num_actions = assistance_game.human_action_space.n
    return np.full((num_states, num_actions), 1 / num_actions)


def functional_random_policy_fn(assistance_game, reward):
    def policy(state):
        return assistance_game.human_action_space.sample()
    return policy


def get_human_policy(assistance_game, reward, max_discount=0.9, num_iter=30, robot_model='optimal', hard=False, **kwargs):
    ag = assistance_game

    value_iteration_fn = hard_value_iteration if hard else soft_value_iteration

    # We want to learn a time independent policy here,
    # so that we get time independent transitions in
    # our assistance problem.
    # So we assume/force the game to be infinite horizon
    # and discounted.
    # This should not be an issue for most environments.
    discount = min(max_discount, ag.discount)

    num_states = ag.state_space.n
    num_actions = ag.human_action_space.n

    # We assume reward depends only on state and actions
    reward = reward.mean(axis=3)

    # Branch for different robot models
    if robot_model == 'random':
        # Human assumes robot acts randomly
        transition = ag.transition.mean(axis=2)
        reward = reward.mean(axis=2)
        policy = value_iteration_fn(transition, reward, discount=discount, num_iter=num_iter)
    elif robot_model == 'optimal':
        # Human assumes robot knows reward
        # and acts optimally
        T = ag.transition
        transition = T.reshape((T.shape[0], -1, T.shape[-1]))
        reward = reward.reshape((T.shape[0], -1))

        full_policy = value_iteration_fn(transition, reward, discount=discount, num_iter=num_iter)
        policy = full_policy.reshape(T.shape[:-1]).sum(axis=2)

    return policy

def hard_value_iteration(T, R, discount=0.9, num_iter=30, **kwargs):
    nS, nA, _ = T.shape
    Q = np.empty((nS, nA))
    V = np.zeros((nS,))

    for _ in range(num_iter):
        Q = R + discount * np.tensordot(T, V, axes=(2, 0))
        V = np.max(Q, axis=1)

    policy = np.eye(nA)[Q.argmax(axis=1)]
    return policy

def soft_value_iteration(T, R, discount=0.9, num_iter=30, beta=1e8, **kwargs):
    nS, nA, _ = T.shape
    Q = np.empty((nS, nA))
    V = np.zeros((nS,))

    for _ in range(num_iter):
        Q = R + discount * np.tensordot(T, V, axes=(2, 0))
        V = logsumexp(beta * Q, axis=1) / beta

    policy = np.exp(beta * (Q - V[:, None]))
    policy /= policy.sum(axis=1, keepdims=True)

    return policy

def softhard_value_iteration(T, R, discount=0.9, num_iter=30, beta=1e8, **kwargs):
    nS, nA, _ = T.shape
    Q = np.empty((nS, nA))
    V = np.zeros((nS,))

    for _ in range(num_iter):
        Q = R + discount * np.tensordot(T, V, axes=(2, 0))
        V = np.max(Q, axis=1)

    policy = np.exp(beta * (Q - V[:, None]))
    policy /= policy.sum(axis=1, keepdims=True)

    return policy
