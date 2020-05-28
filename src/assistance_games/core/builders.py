import functools

import numpy as np
import sparse

from assistance_games.utils import force_sparse

from assistance_games.core.models import (
    BeliefRewardModel,
    DiscreteDistribution,
    TabularBackwardSensorModel,
    TabularForwardSensorModel,
    TabularRewardModel,
    TabularTransitionModel,
)


def discrete_state_space_builder(ag):
    num_rewards = len(ag.reward_distribution)
    num_states = ag.state_space.n * num_rewards
    nS = num_states

    reward_probs = np.array([prob for _, prob in ag.reward_distribution])
    initial_state_distribution = np.einsum('i,j->ij', reward_probs, ag.initial_state_distribution).flatten()

    state_space = DiscreteDistribution(nS, initial_state_distribution)
    return state_space


def get_rewards_and_policies(ag, human_policy_fn):
    # This is being recomputed in multiple builders; if this is slow,
    # we might want to factor it out of the builders, or cache it.
    rewards_and_policies = []
    for rew_idx, (reward, _) in enumerate(ag.reward_distribution):
        kwargs = {'reward_idx': rew_idx}
        rewards_and_policies.append((reward, human_policy_fn(ag, reward, **kwargs)))
    return rewards_and_policies



def tabular_transition_model_fn_builder(ag, human_policy_fn):
    rewards_and_policies = get_rewards_and_policies(ag, human_policy_fn)

    action_space = ag.robot_action_space
    nA = action_space.n

    num_rewards = len(rewards_and_policies)
    num_states = ag.state_space.n * num_rewards
    nS = num_states

    nS0 = ag.state_space.n

    T_shape = (nS, nA, nS)

    is_sparse = isinstance(ag.transition, sparse.COO)

    if not is_sparse:
        T = np.zeros(T_shape)
        for rew_idx, (_, human_policy) in enumerate(rewards_and_policies):
            states_slice = slice(nS0 * rew_idx, nS0 * (rew_idx + 1))
            T[states_slice, :, states_slice] = np.einsum('ij,ijkl->ikl', human_policy, ag.transition)
    else:
        T_coords = [[], [], []]
        T_data = []

        tr = force_sparse(ag.transition)
        ground_states = range(nS0)

        for rew_idx, (_, human_policy) in enumerate(rewards_and_policies):
            lift_state = lambda state : nS0 * rew_idx + state
            human_policy = force_sparse(human_policy)

            # sparse.einsum is not implemented; one alternative is to iterate
            # through ground states instead.
            for s0 in ground_states:
                Ts0 = sparse.tensordot(human_policy[s0], tr[s0], axes=(0, 0))

                state = lift_state(s0)
                actions = Ts0.coords[0]
                next_states = map(lift_state, Ts0.coords[1])

                T_coords[0].extend(state for _ in actions)
                T_coords[1].extend(actions)
                T_coords[2].extend(next_states)
                T_data.extend(Ts0.data)

        T = sparse.COO(T_coords, T_data, T_shape)

    transition_model_fn = functools.partial(TabularTransitionModel, transition_matrix=T)
    return transition_model_fn


def discrete_reward_model_fn_builder(ag, human_policy_fn, use_belief_space=True):
    rewards_and_policies = get_rewards_and_policies(ag, human_policy_fn)
    action_space = ag.robot_action_space
    nA = action_space.n

    num_rewards = len(rewards_and_policies)
    num_states = ag.state_space.n * num_rewards
    nS = num_states

    nS0 = ag.state_space.n

    R_shape = (nS, nA, nS)

    is_sparse = isinstance(ag.transition, sparse.COO)

    if not is_sparse:
        R = np.zeros(R_shape)
        for rew_idx, (reward, human_policy) in enumerate(rewards_and_policies):
            states_slice = slice(nS0 * rew_idx, nS0 * (rew_idx + 1))
            R[states_slice, :, states_slice] = np.einsum('ij,ijkl->ikl', human_policy, reward)
    else:
        R_coords = [[], [], []]
        R_data = []

        tr = force_sparse(ag.transition)
        ground_states = range(nS0)

        for rew_idx, (reward, human_policy) in enumerate(rewards_and_policies):
            lift_state = lambda state : nS0 * rew_idx + state
            reward = force_sparse(reward)
            human_policy = force_sparse(human_policy)

            # sparse.einsum is not implemented; one alternative is to iterate
            # through ground states instead.
            for s0 in ground_states:
                Rs0 = sparse.tensordot(human_policy[s0], reward[s0], axes=(0, 0))

                state = lift_state(s0)
                actions = Rs0.coords[0]
                next_states = map(lift_state, Rs0.coords[1])

                R_coords[0].extend(state for _ in actions)
                R_coords[1].extend(actions)
                R_coords[2].extend(next_states)
                R_data.extend(Rs0.data)

        R = sparse.COO(R_coords, R_data, R_shape)

    reward_model_cls = BeliefRewardModel if use_belief_space else TabularRewardModel
    reward_model_fn = functools.partial(reward_model_cls, reward_matrix=R)
    return reward_model_fn


def forward_sensor_model_fn_builder(ag, human_policy_fn):
    rewards_and_policies = get_rewards_and_policies(ag, human_policy_fn)
    nS0 = ag.state_space.n

    action_space = ag.robot_action_space
    nA = action_space.n

    num_rewards = len(ag.reward_distribution)
    num_states = ag.state_space.n * num_rewards
    nS = num_states

    O_shape = (nA, nS, nS0)

    sensor = np.zeros(O_shape)

    for rew_idx, (reward, human_policy) in enumerate(rewards_and_policies):
        states = range(nS0 * rew_idx, nS0 * (rew_idx + 1))
        ground_states = range(nS0)
        sensor[:, states, ground_states] = 1.0

    sensor_model_fn = functools.partial(TabularForwardSensorModel, sensor=sensor)
    return sensor_model_fn


def back_sensor_model_fn_builder(ag, human_policy_fn):
    rewards_and_policies = get_rewards_and_policies(ag, human_policy_fn)

    nAh = ag.human_action_space.n

    nS0 = ag.state_space.n

    action_space = ag.robot_action_space
    nA = action_space.n

    num_rewards = len(ag.reward_distribution)
    num_states = ag.state_space.n * num_rewards
    nS = num_states

    BO_shape = (nA, nS, nAh)

    back_sensor = np.zeros(BO_shape)
    for rew_idx, (reward, human_policy) in enumerate(rewards_and_policies):
        states = range(nS0 * rew_idx, nS0 * (rew_idx + 1))
        back_sensor[:, states] = human_policy

    sensor_model_fn = functools.partial(TabularBackwardSensorModel, back_sensor=back_sensor)
    return sensor_model_fn
