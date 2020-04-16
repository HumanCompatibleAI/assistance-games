"""Core classes, such as POMDP and AssistanceGame.
"""

import functools

import numpy as np
import gym
from gym.spaces import Discrete, Box
import sparse
from scipy.special import logsumexp

from assistance_games.utils import sample_distribution, uniform_simplex_sample, force_sparse


class POMDP(gym.Env):
    def __init__(
        self,
        state_space,
        sensor_space,
        action_space,
        transition,
        sensor,
        rewards,
        initial_state_distribution,
        initial_belief=None,
        horizon=None,
        back_sensor=None,
        discount=1.0,
    ):
        """Partially Observable Markov Decision Process environment.

        Parameters
        ----------
        state_space : gym.spaces.Discrete, S
        sensor_space : gym.spaces.Discrete, O
        action_space : gym.spaces.Discrete, A
        transition : np.array[|S|, |A|, |S|]
        rewards : np.array[|S|, |A|, |S|]
        sensor : np.array[|A|, |S|, |O|]
        back_sensor : np.array[|A|, |S|, |O'|]
        initial_state_distribution : np.array[|S|]
        initial_belief : np.array[|S|]
        horizon : Float
        discount : Float
        """
        if initial_belief is None:
            initial_belief = initial_state_distribution

        self.state_space = state_space
        self.sensor_space = sensor_space
        self.action_space = action_space
        self.transition = transition
        self.sensor = sensor
        self.back_sensor = back_sensor
        self.rewards = rewards
        self.horizon = horizon
        self.initial_state_distribution = initial_state_distribution
        self.initial_belief = initial_belief
        self.discount = discount
        self.viewer = None

        self.belief_space = Box(low=0.0, high=1.0, shape=(state_space.n,))

    def reset(self):
        self.state = sample_distribution(self.initial_state_distribution)
        self.t = 0
        self.belief = self.initial_belief
        return self.belief

    def step(self, act):
        assert act in self.action_space

        old_state = self.state
        self.state = sample_distribution(self.transition[self.state, act])

        old_belief = self.belief
        ob = self.sample_obs(act, state=old_state, next_state=self.state)
        self.belief = self.update_belief(self.belief, act, ob)

        # Observed reward is myopic
        observed_reward = old_belief @ self.rewards[:, act, :] @ self.belief
        true_reward = self.rewards[old_state, act, self.state]

        self.t += 1
        done = self.horizon is not None and self.t >= self.horizon

        info = {'ob' : ob, 'true_reward' : true_reward}

        return self.belief, observed_reward, done, info
        
    def render(self):
        print(self.state)

    def sample_obs(self, act, state=None, next_state=None):
        if self.back_sensor is not None:
            return sample_distribution(self.back_sensor[act, state])
        else:
            return sample_distribution(self.sensor[act, next_state])

    def update_belief(self, belief, act, ob):
        if self.back_sensor is not None:
            new_belief = (belief * self.back_sensor[act, :, ob]) @ self.transition[:, act, :]
        else:
            new_belief = (belief @ self.transition[:, act, :]) * self.sensor[act, :, ob]
        new_belief /= new_belief.sum()
        return new_belief

    @property
    def observation_space(self):
        return self.belief_space


class AssistanceGame:
    def __init__(
        self,
        state_space,
        human_action_space,
        robot_action_space,
        transition,
        reward_distribution,
        initial_state_distribution,
        horizon=None,
        discount=1.0,
    ):
        """Two-agent MDP, with shared reward hidden from second agent.

        Parameters
        ----------
        state_space : gym.spaces.Discrete, S
        human_action_space : gym.spaces.Discrete, A_h
        robot_action_space : gym.spaces.Discrete, A_r
        transition : np.array[|S|, |A_h|, |A_r|, |S|]
        reward_distribution : List[Tuple[np.array[|S|, |A_h|, |A_r|, |S|], Float]]
        initial_state_distribution : np.array[|S|]
        horizon : Float
        discount : Float
        """
        self.state_space = state_space
        self.human_action_space = human_action_space
        self.robot_action_space = robot_action_space
        self.transition = transition
        self.reward_distribution = reward_distribution
        self.initial_state_distribution = initial_state_distribution
        self.horizon = horizon
        self.discount = discount


class AssistanceProblem(POMDP):
    def __init__(self, assistance_game, human_policy_fn, is_sparse=True, define_sensor=False):
        """
        Parameters
        ----------
        assistance_game : AssistanceGame
        human_policy_fn : AssistanceGame -> Reward (np.array[|S|, |A_h|, |A_r|, |S|])
                                         -> Policy (np.array[|S|, |A_h|])
        is_sparse : Bool
            Whether the transition and reward matrices are to be sparse.

        For each possible reward, we compute a human policy, and thus
        we compute the corresponding transition and reward matrices
        for each (state, robot_action, next_state) tuple.
        """
        ag = assistance_game
        nAh = ag.human_action_space.n

        sensor_space = ag.state_space
        nS0 = ag.state_space.n

        action_space = ag.robot_action_space
        nA = action_space.n

        num_rewards = len(ag.reward_distribution)
        num_states = ag.state_space.n * num_rewards
        state_space = Discrete(num_states)
        nS = state_space.n

        O_shape = (nA, nS, nS0)
        BO_shape = (nA, nS, nAh)
        T_shape = (nS, nA, nS)
        R_shape = (nS, nA, nS)

        if define_sensor:
            sensor = np.zeros(O_shape)
        else:
            sensor = None

        back_sensor = np.zeros(BO_shape)
        if not is_sparse:
            transition = np.zeros(T_shape)
            rewards = np.zeros(R_shape)

            for rew_idx, (reward, _) in enumerate(ag.reward_distribution):
                human_policy = human_policy_fn(assistance_game, reward)

                states_slice = slice(nS0 * rew_idx, nS0 * (rew_idx + 1))
                states = range(nS0 * rew_idx, nS0 * (rew_idx + 1))
                ground_states = range(nS0)

                transition[states_slice, :, states_slice] = np.einsum('ij,ijkl->ikl', human_policy, ag.transition)
                rewards[states_slice, :, states_slice] = np.einsum('ij,ijkl->ikl', human_policy, reward)

                if define_sensor:
                    sensor[:, states, ground_states] = 1.0
                back_sensor[:, states] = human_policy
        else:
            # This should be doing the exact same thing as the other
            # branch, but here T and R are sparse matrices.
            T_coords = [[], [], []]
            T_data = []
            R_coords = [[], [], []]
            R_data = []

            tr = force_sparse(ag.transition)

            for rew_idx, (reward, _) in enumerate(ag.reward_distribution):
                human_policy = human_policy_fn(assistance_game, reward)
                ground_states = range(nS0)
                states = range(nS0 * rew_idx, nS0 * (rew_idx + 1))
                lift_state = lambda state : nS0 * rew_idx + state

                reward = force_sparse(reward)
                human_policy_sparse = force_sparse(human_policy)

                # sparse.einsum is not implemented; thus we have to iterate
                # through states with a for loop.
                for s0 in ground_states:
                    T0 = sparse.tensordot(human_policy_sparse[s0], tr[s0], axes=(0, 0))
                    R0 = sparse.tensordot(human_policy_sparse[s0], reward[s0], axes=(0, 0))

                    T_coords[0].extend(lift_state(s0) for _ in T0.coords[0])
                    T_coords[1].extend(T0.coords[0])
                    T_coords[2].extend(map(lift_state, T0.coords[1]))
                    T_data.extend(T0.data)

                    R_coords[0].extend(lift_state(s0) for _ in R0.coords[0])
                    R_coords[1].extend(R0.coords[0])
                    R_coords[2].extend(map(lift_state, R0.coords[1]))
                    R_data.extend(R0.data)

                if define_sensor:
                    sensor[:, states, ground_states] = 1.0
                back_sensor[:, states] = human_policy

            transition = sparse.COO(T_coords, T_data, T_shape)
            rewards = sparse.COO(R_coords, R_data, R_shape)


        reward_probs = np.array([prob for _, prob in ag.reward_distribution])
        initial_state_distribution = np.einsum('i,j->ij', reward_probs, ag.initial_state_distribution).flatten()

        discount = ag.discount
        horizon = ag.horizon

        super().__init__(
            state_space=state_space,
            sensor_space=sensor_space,
            action_space=action_space,
            transition=transition,
            sensor=sensor,
            back_sensor=back_sensor,
            rewards=rewards,
            horizon=horizon,
            initial_state_distribution=initial_state_distribution,
            discount=discount,
        )
        self.num_obs = ag.state_space.n
        self.num_rewards = num_rewards


class POMDPPolicy:
    """Policy from alpha vectors provided by POMDP solvers"""
    def __init__(self, alphas):
        self.alpha_vectors = []
        self.alpha_actions = []
        for vec, act in alphas:
            self.alpha_vectors.append(vec)
            self.alpha_actions.append(act)

    def predict(self, belief, state=None, deterministic=True):
        idx = np.argmax(self.alpha_vectors @ belief)
        return self.alpha_actions[idx], state


### Human Policies

def random_policy_fn(assistance_game, reward):
    num_states = assistance_game.state_space.n
    num_actions = assistance_game.human_action_space.n
    return np.full((num_states, num_actions), 1 / num_actions)


def get_human_policy(assistance_game, reward, max_discount=0.9, num_iter=30, robot_model='optimal', hard=False):
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

def hard_value_iteration(T, R, discount=0.9, num_iter=30):
    nS, nA, _ = T.shape
    Q = np.empty((nS, nA))
    V = np.zeros((nS,))

    for _ in range(num_iter):
        Q = R + discount * np.tensordot(T, V, axes=(2, 0))
        V = np.max(Q, axis=1)

    policy = np.eye(nA)[Q.argmax(axis=1)]
    return policy

def soft_value_iteration(T, R, discount=0.9, num_iter=30, beta=1e8):
    nS, nA, _ = T.shape
    Q = np.empty((nS, nA))
    V = np.zeros((nS,))

    for _ in range(num_iter):
        Q = R + discount * np.tensordot(T, V, axes=(2, 0))
        V = logsumexp(beta * Q, axis=1) / beta

    policy = np.exp(beta * (Q - V[:, None]))
    policy /= policy.sum(axis=1, keepdims=True)

    return policy

