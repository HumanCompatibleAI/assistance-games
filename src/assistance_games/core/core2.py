from abc import ABC, abstractmethod
import gym
import numpy as np


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

    def get_obs(self, state):
        return state

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
        return self.get_obs(self.state)

    def step(self, action):
        info = {}

        # Update state
        prev_state = self.state
        next_state = self.sample_transition(prev_state, action)
        self.state = next_state
        self.t += 1

        reward = self.get_reward(prev_state, action, next_state)

        # Update sensor
        # self.sense = self.sample_sense(prev_state, action, next_state)

        # Update observation
        ob = self.get_obs(next_state)

        # Check termination
        done = self.is_terminal(next_state) or self.t >= self.horizon

        return ob, reward, done, info


class AssistancePOMDP(ABC):
    def __init__(self, discount, horizon, theta_dist, init_state_dist, observation_space=None, action_space=None, default_aH=None, default_aR=None):
        self.discount = discount
        self.horizon = horizon
        self.theta_distribution = theta_dist
        self.initial_state_distribution = init_state_dist
        self.observation_space = observation_space
        self.action_space = action_space
        self.default_aH = default_aH
        self.default_aR = default_aR
    
    def get_obs_human(self, state):
        return state

    def get_obs_robot(self, state):
        return state

    def encode_obs(self, obs, prev_aH):
        return (obs, prev_aH)

    @abstractmethod
    def get_transition_distribution(self, state, aH, aR):
        pass

    @abstractmethod
    def get_reward(self, state, aH, aR, next_state, theta):
        pass

    @abstractmethod
    def get_human_action_distribution(self, obsH, prev_aR, theta):
        pass

    @abstractmethod
    def is_terminal(self, state):
        pass

    @abstractmethod
    def render(self, state, theta, mode='human'):
        pass


class ReducedAssistancePOMDP(POMDP):
    def __init__(self, apomdp):
        super().__init__(
            discount=apomdp.discount,
            horizon=apomdp.horizon,
            init_state_dist=ExtendedInitialStateDistribution(apomdp),
            observation_space=apomdp.observation_space,
            action_space=apomdp.action_space
        )
        self.apomdp = apomdp

    def get_obs(self, state):
        s, next_aH, prev_aH, theta = state
        return self.apomdp.encode_obs(self.apomdp.get_obs_robot(s), prev_aH)

    def get_transition_distribution(self, state, action):
        s, next_aH, prev_aH, theta = state
        underlying_transition = self.apomdp.get_transition_distribution(s, next_aH, action)
        return ExtendedTransitionDistribution(underlying_transition, next_aH, action, theta, self.apomdp)

    def get_reward(self, state, action, next_state):
        s, next_aH, prev_aH, theta = state
        s2, _, _, _ = next_state
        return self.apomdp.get_reward(s, next_aH, action, s2, theta)

    def is_terminal(self, state):
        s, next_aH, prev_aH, theta = state
        return self.apomdp.is_terminal(s)

    def render(self, mode='human'):
        s, next_aH, prev_aH, theta = self.state
        self.apomdp.render(s, theta, mode=mode)


class Distribution(ABC):
    @abstractmethod
    def support(self):
        pass

    @abstractmethod
    def get_probability(self, x):
        pass

    def sample(self):
        sample = np.random.random()
        for x in self.support():
            total_prob += self.get_probability(x)
            if total_prob >= sample:
                return x

        raise ValueError("Total probability was less than 1")


class ContinuousDistribution(Distribution):
    def support(self):
        raise ValueError("Cannot ask for support of a continuous distribution")

    def get_probability(self, x):
        raise ValueError("Cannot get probability of an element of a continuous distribution")


class KroneckerDistribution(Distribution):
    def __init__(self, x):
        self.x = x

    def support(self):
        yield x

    def get_probability(self, x):
        assert x == self.x
        return 1.0

    def sample(self):
        return self.x


class DictionaryDistribution(Distribution):
    def __init__(self, x_to_prob_dict):
        self.x_to_prob_dict = x_to_prob_dict

    def support(self):
        for x in self.x_to_prob_dict.keys():
            yield x

    def get_probability(self, x):
        return self.x_to_prob_dict[x]


class UniformDiscreteDistribution(Distribution):
    def __init__(self, options):
        self.options = list(options)

    def support(self):
        for x in self.options:
            yield x

    def get_probability(self, x):
        assert x in self.options
        return 1.0 / len(self.options)

    def sample(self):
        return np.random.choice(self.options)


class UniformContinuousDistribution(ContinuousDistribution):
    def __init__(self, lows, highs):
        self.lows = lows
        self.highs = highs

    def sample(self):
        return np.random.uniform(self.lows, self.highs)


class ExtendedTransitionDistribution(Distribution):
    def __init__(self, underlying_transition_distribution, prev_aH, prev_aR, theta, apomdp):
        self.trans_dist = underlying_transition_distribution
        self.prev_aH = prev_aH
        self.prev_aR = prev_aR
        self.theta = theta
        self.apomdp = apomdp

    def support(self):
        for next_state in self.trans_dist.support():
            obsH = self.apomdp.get_obs_human(next_state)
            policy = self.apomdp.get_human_action_distribution(obsH, self.prev_aR, self.theta)
            for next_aH in policy.support():
                yield (next_state, next_aH, self.prev_aH, self.theta)

    def get_probability(self, x):
        next_state, next_aH, prev_aH, theta = x
        assert prev_aH == self.prev_aH
        assert theta == self.theta
        obsH = self.apomdp.get_obs_human(next_state)
        policy = self.apomdp.get_human_action_distribution(obsH, self.prev_aR, self.theta)
        return self.trans_dist.get_probability(next_state) * policy.get_probability(next_aH)

    def sample(self):
        next_state = self.trans_dist.sample()
        obsH = self.apomdp.get_obs_human(next_state)
        policy = self.apomdp.get_human_action_distribution(obsH, self.prev_aR, self.theta)
        next_aH = policy.sample()
        return next_state, next_aH, self.prev_aH, self.theta
        

class ExtendedInitialStateDistribution(Distribution):
    def __init__(self, apomdp):
        self.apomdp = apomdp

    def support(self):
        prev_aH, prev_aR = self.apomdp.default_aH, self.apomdp.default_aR
        for state in self.apomdp.initial_state_distribution.support():
            for theta in self.apomdp.theta_distribution.support():
                obsH = self.apomdp.get_obs_human(state)
                policy = self.apomdp.get_human_action_distribution(obsH, prev_aR, theta)
                for next_aH in policy.support():
                    yield state, next_aH, prev_aH, theta

    def get_probability(self, x):
        state, next_aH, prev_aH, theta = x
        assert prev_aH == self.apomdp.default_aH
        obsH = self.apomdp.get_obs_human(state)
        policy = self.apomdp.get_human_action_distribution(obsH, self.apomdp.default_aR, theta)
        result = policy.get_probability(next_aH)
        result *= self.apomdp.theta_distribution.get_probability(theta)
        result *= self.apomdp.initial_state_distribution.get_probability(state)
        return result

    def sample(self):
        state = self.apomdp.initial_state_distribution.sample()
        theta = self.apomdp.theta_distribution.sample()
        obsH = self.apomdp.get_obs_human(state)
        policy = self.apomdp.get_human_action_distribution(obsH, self.apomdp.default_aR, theta)
        next_aH = policy.sample()
        prev_aH = self.apomdp.default_aH
        return state, next_aH, prev_aH, theta

