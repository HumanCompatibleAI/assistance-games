import gym
import numpy as np

from assistance_games.core.distributions import KroneckerDistribution, DiscreteDistribution

class POMDP(gym.Env):
    """POMDP class.

    Note: All states are assumed to be immutable.
    """
    def __init__(self, discount, horizon, init_state_dist, observation_space, action_space):
        self.discount = discount
        self.horizon = horizon
        self.initial_state_distribution = init_state_dist
        self.observation_space = observation_space
        self.action_space = action_space

    def get_obs_distribution(self, state):
        return KroneckerDistribution(state)

    def get_transition_distribution(self, state, action):
        raise NotImplementedError

    def sample_transition(self, state, action):
        return self.get_transition_distribution(state, action).sample()

    def get_reward(self, state, action, next_state):
        raise NotImplementedError

    def is_terminal(self, state):
        raise NotImplementedError

    def render(self, mode='human'):
        raise NotImplementedError

    def reset(self):
        self.state = self.initial_state_distribution.sample()
        self.t = 0
        return self.get_obs_distribution(self.state).sample()

    def step(self, action):
        prev_state = self.state
        next_state = self.sample_transition(prev_state, action)
        self.state = next_state
        self.t += 1

        ob = self.get_obs_distribution(next_state).sample()
        reward = self.get_reward(prev_state, action, next_state)
        done = self.is_terminal(next_state) or self.t >= self.horizon
        info = {}

        return ob, reward, done, info


class POMDPWithMatrices(POMDP):
    """A POMDP with all the elements needed to run value iteration or PBVI."""
    def __init__(self, T, R, O, discount, horizon, init_state_dist):
        self.nS, self.nA, _ = T.shape
        _, self.nO = O.shape
        assert T.shape == (self.nS, self.nA, self.nS)
        assert R.shape == (self.nS, self.nA, self.nS)
        assert O.shape == (self.nS, self.nO)

        observation_space = gym.spaces.Discrete(self.nO)
        action_space = gym.spaces.Discrete(self.nA)
        super().__init__(discount, horizon, init_state_dist, observation_space, action_space)

        self._T = T
        self._R = R
        self._O = O

    def get_obs_distribution(self, state):
        return DiscreteDistribution(self._O[state, :])

    def get_transition_distribution(self, state, action):
        return DiscreteDistribution(self._T[state, action, :])

    def get_reward(self, state, action, next_state):
        return self._R[state, action, next_state]

    def is_terminal(self, state):
        return False

    def get_num_states(self):
        return self.nS

    def get_num_actions(self):
        return self.nA

    def get_transition_matrix(self):
        """Returns the transition matrix T (Numpy array or sparse matrix).

        T[s1, a, s2] is the probability of transition to s2 when taking action a in s1.
        """
        return self._T

    def get_reward_matrix(self):
        """Returns the reward model R (Numpy array or sparse matrix).

        R[s1, a, s2] is the reward for transitioning to s2 when taking action a in s1.
        """
        return self._R

    def get_observation_matrix(self):
        """Returns the observation model O (Numpy array or sparse matrix).

        O[s, o] is the probability of observation o in state s.
        """
        return self._O

    def numpy_initial_state_distribution(self):
        dist = np.zeros(self.get_num_states())
        for s in self.initial_state_distribution.support():
            dist[s] = self.initial_state_distribution.get_probability(s)
        return dist
