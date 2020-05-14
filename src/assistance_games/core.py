"""Core classes, such as POMDP and AssistanceGame.
"""

import functools

import numpy as np
import gym
from gym.spaces import Discrete, MultiDiscrete, Box
import sparse
from scipy.special import logsumexp

from assistance_games.utils import sample_distribution, uniform_simplex_sample, force_sparse

class DiscreteDistribution(Discrete):
    def __init__(self, n, p=None):
        if p is None:
            p = (1/n) * np.ones(n)

        super().__init__(n)
        self.p = p

    def sample_initial_state(self):
        return sample_distribution(self.p)

    def distribution(self):
        return self.p


##### Begin models


### Transition models

class TransitionModel:
    def __init__(self, pomdp):
        self.pomdp = pomdp

    def __call__(self):
        pass

class TabularTransitionModel(TransitionModel):
    def __init__(self, pomdp, transition_matrix):
        super().__init__(pomdp)
        self.transition_matrix = transition_matrix

    @property
    def T(self):
        return self.transition_matrix

    def __call__(self):
        s = self.pomdp.state
        a = self.pomdp.action
        return sample_distribution(self.T[s, a])

    def transition_belief(self, belief, action):
        return belief @ self.T[:, action, :]


### Observation models

class ObservationModel:
    def __init__(self, pomdp):
        self.pomdp = pomdp

    def __call__(self):
        pass

class BeliefObservationModel(ObservationModel):
    def __init__(self, pomdp):
        super().__init__(pomdp)
        self.belief = None
        self.prev_belief = None

    def __call__(self):
        self.prev_belief = self.belief
        if self.pomdp.t == 0:
            self.belief = self.pomdp.state_space.distribution()
        else:
            self.belief = self.pomdp.sensor_model.update_belief(self.belief)
        return self.belief

    @property
    def space(self):
        return Box(low=0.0, high=1.0, shape=(self.pomdp.state_space.n,))

class SenseObservationModel(ObservationModel):
    def __call__(self):
        return self.sensor_model.sense

    @property
    def space(self):
        return self.pomdp.sensor_model.space

class FeatureSenseObservationModel(ObservationModel):
    def __init__(self, pomdp, feature_extractor):
        super().__init__(pomdp)
        self.feature_extractor = feature_extractor

    def __call__(self):
        feature = self.feature_extractor(self.pomdp.state)

        sense = self.pomdp.sensor_model.sense
        if sense is None:
            sense = self.pomdp.sensor_model.space.sample()
        obs = np.zeros((len(feature) + self.pomdp.sensor_model.space.n, 1))
        obs[:len(feature), 0] = feature
        obs[len(feature) + sense, 0] = 1
        return obs # np.array([feature, sense])

    @property
    def space(self):
        num_senses = self.pomdp.sensor_model.space.n
        num_features = self.feature_extractor.n
        return Box(low=0.0, high=self.pomdp.assistance_game.max_feature_value, shape=(num_features + num_senses, 1))
        # MultiDiscrete([num_features, num_senses])
        #return Discrete(num_features + num_senses)


### Sensor models

class SensorModel:
    def __init__(self, pomdp):
        self.pomdp = pomdp

    def __call__(self):
        pass

class TabularForwardSensorModel(SensorModel):
    def __init__(self, pomdp, sensor):
        self.pomdp = pomdp
        self.sensor = sensor
        self.sense = None

    def __call__(self):
        return self.sample_sense(state=self.pomdp.prev_state, action=self.pomdp.action, next_state=self.pomdp.state)

    def sample_sense(self, *, state=None, action=None, next_state=None):
        self.sense = sample_distribution(self.sensor[action, next_state])
        return self.sense

    def update_belief(self, belief, action=None, sense=None):
        if action is None:
            action = self.pomdp.action
        if sense is None:
            sense = self.sense

        new_belief = self.pomdp.transition_model.transition_belief(belief, action=action) * self.sensor[action, :, sense]
        new_belief /= new_belief.sum()
        return new_belief

    @property
    def space(self):
        num_senses = self.sensor.shape[-1]
        return Discrete(num_senses)


class TabularBackwardSensorModel(SensorModel):
    def __init__(self, pomdp, back_sensor):
        self.pomdp = pomdp
        self.back_sensor = back_sensor
        self.sense = None

    def __call__(self):
        return self.sample_sense(state=self.pomdp.prev_state, action=self.pomdp.action, next_state=self.pomdp.state)

    def sample_sense(self, *, state=None, action=None, next_state=None):
        self.sense = sample_distribution(self.back_sensor[action, state])
        return self.sense

    def update_belief(self, belief, action=None, sense=None):
        if action is None:
            action = self.pomdp.action
        if sense is None:
            sense = self.sense

        new_belief = self.pomdp.transition_model.transition_belief(belief * self.back_sensor[action, :, sense], action=action)
        new_belief /= new_belief.sum()
        return new_belief

    @property
    def space(self):
        num_senses = self.back_sensor.shape[-1]
        return Discrete(num_senses)


### Reward models

class RewardModel:
    def __init__(self, pomdp):
        self.pomdp = pomdp

    def __call__(self):
        pass

class TabularRewardModel(RewardModel):
    def __init__(self, pomdp, reward_matrix):
        super().__init__(pomdp)
        self.reward_matrix = reward_matrix

    @property
    def R(self):
        return self.reward_matrix

    def __call__(self):
        prev_state = self.pomdp.prev_state
        action = self.pomdp.action
        state = self.pomdp.state
        return self.R[prev_state, action, state]

class BeliefRewardModel(RewardModel):
    def __init__(self, pomdp, reward_matrix):
        super().__init__(pomdp)
        self.reward_matrix = reward_matrix

    @property
    def R(self):
        return self.reward_matrix

    def __call__(self):
        prev_belief = self.pomdp.observation_model.prev_belief
        action = self.pomdp.action
        belief = self.pomdp.observation_model.belief
        return prev_belief @ self.R[:, action, :] @ belief


### Termination models

class TerminationModel:
    def __init__(self, pomdp):
        self.pomdp = pomdp

    def __call__(self):
        pass


class NoTerminationModel(TerminationModel):
    def __call__(self):
        return False


##### End models


class POMDP(gym.Env):
    def __init__(
        self,
        *,
        state_space,
        action_space,
        observation_model_fn=BeliefObservationModel,
        transition_model_fn=None,
        sensor_model_fn=None,
        reward_model_fn=None,
        termination_model_fn=NoTerminationModel,
        horizon=np.inf,
        discount=1.0,
    ):
        self.state_space = state_space
        self.action_space = action_space
        self.horizon = horizon
        self.discount = discount

        self.transition_model = transition_model_fn(self)
        self.observation_model = observation_model_fn(self)
        self.sensor_model = sensor_model_fn(self)
        self.reward_model = reward_model_fn(self)
        self.termination_model = termination_model_fn(self)

        self.viewer = None

    def reset(self):
        self.prev_state = None
        self.state = self.state_space.sample_initial_state()
        self.t = 0
        return self.observation_model()

    def step(self, action):
        self.action = action

        self.info = {}

        # Update state
        self.prev_state = self.state
        self.state = self.transition_model()
        self.t += 1

        # Update sensor
        sense = self.sensor_model()

        # Update observation
        ob = self.observation_model()

        # Compute reward
        reward = self.reward_model()

        # Check termination
        done = self.termination_model() or self.t >= self.horizon

        return ob, reward, done, self.info

    @property
    def observation_space(self):
        return self.observation_model.space


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
        """
        self.state_space = state_space
        self.human_action_space = human_action_space
        self.robot_action_space = robot_action_space
        self.transition = transition
        self.reward_distribution = reward_distribution
        self.initial_state_distribution = initial_state_distribution
        self.horizon = horizon
        self.discount = discount


##### Model builders

def tabular_transition_model_fn_builder(ag, human_policy_fn):
    # This is being recomputed in multiple builders; if this is slow,
    # we might want to factor it out, or cache it.
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

def state_space_builder(ag):
    num_rewards = len(ag.reward_distribution)
    num_states = ag.state_space.n * num_rewards
    nS = num_states

    reward_probs = np.array([prob for _, prob in ag.reward_distribution])
    initial_state_distribution = np.einsum('i,j->ij', reward_probs, ag.initial_state_distribution).flatten()

    state_space = DiscreteDistribution(nS, initial_state_distribution)
    return state_space

##### End model builders


class AssistanceProblem(POMDP):
    def __init__(
        self,
        assistance_game,
        human_policy_fn,
        transition_model_fn_builder=tabular_transition_model_fn_builder,
        reward_model_fn_builder=discrete_reward_model_fn_builder,
        sensor_model_fn_builder=back_sensor_model_fn_builder,
        observation_model_fn=BeliefObservationModel,
    ):
        """
        Parameters
        ----------
        assistance_game : AssistanceGame
        human_policy_fn : AssistanceGame -> Reward -> Policy
        """
        ag = assistance_game

        super().__init__(
            state_space=state_space_builder(ag),
            action_space=ag.robot_action_space,
            horizon=ag.horizon,
            discount=ag.discount,

            transition_model_fn=transition_model_fn_builder(ag, human_policy_fn),
            reward_model_fn=reward_model_fn_builder(ag, human_policy_fn),
            sensor_model_fn=sensor_model_fn_builder(ag, human_policy_fn),
            observation_model_fn=observation_model_fn,
        )



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


def get_rewards_and_policies(ag, human_policy_fn):
    rewards_and_policies = []
    for rew_idx, (reward, _) in enumerate(ag.reward_distribution):
        kwargs = {'reward_idx': rew_idx}
        rewards_and_policies.append((reward, human_policy_fn(ag, reward, **kwargs)))
    return rewards_and_policies


### Human Policies

def random_policy_fn(assistance_game, reward):
    num_states = assistance_game.state_space.n
    num_actions = assistance_game.human_action_space.n
    return np.full((num_states, num_actions), 1 / num_actions)


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
