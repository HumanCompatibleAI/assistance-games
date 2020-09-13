from abc import ABC, abstractmethod
import gym
import numpy as np
from assistance_games.core.models import TabularForwardSensorModel
from assistance_games.utils import dict_to_sparse


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
        prev_state = self.state
        next_state = self.sample_transition(prev_state, action)
        self.state = next_state
        self.t += 1

        ob = self.get_obs(next_state)
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
        self.sensor_model = TabularForwardSensorModel(self, self._O)

    def get_obs(self, state):
        return DiscreteDistribution(self._O[state, :]).sample()

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

    def numpy_initial_state_distribution(self):
        dist = np.zeros(self.get_num_states())
        for s in self.initial_state_distribution.support():
            dist[s] = self.initial_state_distribution.get_probability(s)
        return dist


class AssistancePOMDP(ABC):
    def __init__(self, discount, horizon, theta_dist, init_state_dist,
                 observation_space=None, action_space=None,
                 default_aH=None, default_aR=None,
                 deterministic=False, fully_observable=False):
        self.discount = discount
        self.horizon = horizon
        self.theta_distribution = theta_dist
        self.initial_state_distribution = init_state_dist
        self.observation_space = observation_space
        self.action_space = action_space
        self.default_aH = default_aH
        self.default_aR = default_aR
        self.deterministic = deterministic
        self.fully_observable = fully_observable
    
    def get_obs_human(self, state):
        # Assume full observability by default
        return state

    def get_obs_robot(self, state):
        # Assume full observability by default
        return state

    def encode_obs(self, obs, prev_aH):
        """Once converted to a POMDP, observations are tuples (obs,
        prev_aH). This method allows postprocessing, e.g. to make it suitable
        for input to a policy net.
        """
        return (obs, prev_aH)

    @abstractmethod
    def get_transition_distribution(self, state, aH, aR):
        """Returns the distribution T( . | state, aH, aR)."""
        pass

    @abstractmethod
    def get_reward(self, state, aH, aR, next_state, theta):
        """Returns the reward for transition (state, aH, aR, next_state)."""
        pass

    @abstractmethod
    def get_human_action_distribution(self, obsH, prev_aR, theta):
        """Returns the human policy PiH( . | obsH, prev_aR, theta)."""
        pass

    @abstractmethod
    def is_terminal(self, state):
        """Returns True if state is terminal, False otherwise."""
        pass

    @abstractmethod
    def render(self, state, prev_aH, prev_aR, theta, mode='human'):
        pass

    def close(self):
        pass


class AssistancePOMDPWithMatrixSupport(AssistancePOMDP):
    """When extending this class, in addition to implementing the methods here
    and in AssistancePOMDP, make sure to set self.nS, self.nAH, self.nAR, and
    self.nOR.

    The main additions are the ability to convert between whatever state
    representation is used in the underlying environment, and numerical IDs for
    states (0, 1, ... nS - 1). Similarly for actions and robot observations.

    In addition, it is possible to automatically compute a human
    policy using value iteration if the environment is fully
    observable. This can be done by specifying a dictionary for
    human_policy_type with keys 'H' and 'R', specifying the models for
    each. Currently, H must be 'optimal', and R can be either 'random'
    or 'optimal'. The key 'num_iters' is optional, if present it
    specifies the number of iterations of value iteration to run
    (default value 30). When H is 'noisy', optionally 'beta' may be
    provided (default value 1).
    """
    def __init__(self, human_policy_type=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.thetas = list(self.theta_distribution.support())
        self.theta_map = {theta:i for i, theta in enumerate(self.thetas)}
        if human_policy_type is not None:
            print('Computing human policy using value iteration')
            self._compute_human_policy(human_policy_type)

    def index_to_state(self, num):
        """Convert numeric state to underlying env state."""
        return num

    def state_to_index(self, state):
        """Convert underlying env state to numeric state."""
        return state

    def index_to_human_action(self, num):
        return num

    def human_action_to_index(self, aH):
        return aH

    def index_to_robot_action(self, num):
        return num

    def robot_action_to_index(self, aR):
        return aR

    # These should be overridden if the environment is not fully observable
    def index_to_robot_obs(self, num):
        assert self.fully_observable
        return self.index_to_state(num)

    def robot_obs_to_index(self, oR):
        assert self.fully_observable
        return self.state_to_index(oR)

    def get_human_action_distribution(self, obsH, prev_aR, theta):
        new_s = self.theta_map[theta] * self.nS + self.state_to_index(obsH)
        probs = self.human_policy[new_s, :]
        return DiscreteDistribution(dict(zip(range(self.nAH), probs)))

    def _make_T_and_R_matrices(self):
        # Wasteful, duplicates theta -- could save on space by having
        # T and R with dimensions [nThetas, nS, nAH, nAR, nS] and
        # modifying the value iteration algorithms to handle this
        assert self.fully_observable
        new_nS = len(self.thetas) * self.nS
        T, R = {}, {}
        for s, aH, aR, s2, theta, r, p in self.enumerate_transitions():
            new_s = theta * self.nS + s
            new_s2 = theta * self.nS + s2
            T[new_s, aH, aR, new_s2] = p
            R[new_s, aH, aR, new_s2] = r
        T = dict_to_sparse(T, (new_nS, self.nAH, self.nAR, new_nS))
        R = dict_to_sparse(R, (new_nS, self.nAH, self.nAR, new_nS))
        return T, R

    def _compute_human_policy(self, type):
        """See class docstring for the meaning of `type`."""
        assert self.fully_observable
        # TODO: This seems wasteful, we already make matrices in the
        # reduction, presumably we could reuse those
        T, R = self._make_T_and_R_matrices()
        num_iters = type.get('num_iters', 30)
        assert type['R'] in ['random', 'optimal']
        if type['R'] == 'random':
            # Human assumes robot acts randomly
            T = T.mean(axis=2)
            R = R.mean(axis=2)
            self.human_policy = value_iteration(T, R, discount=self.discount, num_iters=num_iters)
        else:
            # Human assumes robot knows reward and acts optimally, so
            # treat robot action space as part of human action space
            # to compute a joint policy
            nS, nAH, nAR, _ = T.shape
            T = T.reshape((nS, nAH * nAR, nS))
            R = R.reshape((nS, nAH * nAR, nS))
            joint_policy = value_iteration(T, R, discount=self.discount, num_iters=num_iters)
            # Human policy just marginalizes out the robot action
            self.human_policy = joint_policy.reshape((nS, nAH, nAR)).sum(axis=2)

    def enumerate_transitions(self):
        """Return a sequence of (s, aH, aR, s', theta, r, p) transitions.

        Sequence can contain elements with p = 0, and must contain all
        transitions with p > 0.

        When subclassing this class, you do not need to override this method,
        but you may wish to do so for improved efficiency.
        """
        for theta in range(len(self.thetas)):
            theta_real = self.thetas[theta]
            for s in range(self.nS):
                s_real = self.index_to_state(s)
                for aH in range(self.nAH):
                    aH_real = self.index_to_human_action(aH)
                    for aR in range(self.nAR):
                        aR_real = self.index_to_robot_action(aR)

                        t = self.get_transition_distribution(s_real, aH_real, aR_real)
                        for s2_real in t.support():
                            p = t.get_probability(s2_real)
                            r = self.get_reward(s_real, aH_real, aR_real, s2_real, theta_real)
                            s2 = self.state_to_index(s2_real)
                            yield (s, aH, aR, s2, theta, r, p)


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

    def render(self, mode='human', prev_action=None):
        s, next_aH, prev_aH, theta = self.state
        self.apomdp.render(s, prev_aH, prev_action, theta, mode=mode)

    def close(self):
        self.apomdp.close()


class ReducedAssistancePOMDPWithMatrices(ReducedAssistancePOMDP):
    """Creates matrices required to run value iteration based methods.

    This requires states and actions to be indices, and so it assumes that the
    underlying Assistance POMDP provides methods to do this conversion. In
    addition, overrides step and reset to convert numeric states and actions
    (which the solver assumes) into structured versions (which the underlying
    environment assumes).

    Note: We could inherit from POMDPWithMatrices, but we choose to rely on duck
    typing rather than deal with multiple inheritance.
    """
    def __init__(self, apomdp):
        assert hasattr(apomdp, 'nS') and hasattr(apomdp, 'nOR') and hasattr(apomdp, 'nAH') and hasattr(apomdp, 'nAR')
        super().__init__(apomdp)
        print('Creating T, R and O matrices')
        self._create_matrices()

    def _create_matrices(self):
        self.nS = self.apomdp.nS * self.apomdp.nAH * self.apomdp.nAH * len(self.apomdp.thetas)
        self.nA = self.apomdp.nAR
        self.nO = self.apomdp.nOR * self.apomdp.nAH

        T_dict, R_dict, O_dict = {}, {}, {}
        for s, aH, aR, s2, theta, r, p in self.apomdp.enumerate_transitions():
            s_real = self.apomdp.index_to_state(s)
            aH_real = self.apomdp.index_to_human_action(aH)
            aR_real = self.apomdp.index_to_robot_action(aR)
            s2_real = self.apomdp.index_to_state(s2)
            theta_real = self.apomdp.thetas[theta]

            for prev_aH in range(self.apomdp.nAH):
                new_s = self.indexify_state(s, aH, prev_aH, theta)
                obsR_real = self.apomdp.get_obs_robot(s_real)
                obsR = self.apomdp.robot_obs_to_index(obsR_real)
                new_obs = self.indexify_obs(obsR, prev_aH)
                O_dict[new_s, new_obs] = 1.0

                obsH_real = self.apomdp.get_obs_human(s2_real)
                policy = self.apomdp.get_human_action_distribution(obsH_real, aR_real, theta_real)
                for next_aH_real in policy.support():
                    p_next_aH = policy.get_probability(next_aH_real)
                    next_aH = self.apomdp.human_action_to_index(next_aH_real)

                    new_s2 = self.indexify_state(s2, next_aH, aH, theta)
                    T_dict[new_s, aR, new_s2] = p * p_next_aH
                    R_dict[new_s, aR, new_s2] = r

        self._T = dict_to_sparse(T_dict, (self.nS, self.nA, self.nS))
        self._R = dict_to_sparse(R_dict, (self.nS, self.nA, self.nS))
        self._O = dict_to_sparse(O_dict, (self.nS, self.nO))
        self.sensor_model = TabularForwardSensorModel(self, self._O)

    def indexify_state(self, s, next_aH, prev_aH, theta):
        result = theta
        result = result * self.apomdp.nS + s
        result = result * self.apomdp.nAH + next_aH
        result = result * self.apomdp.nAH + prev_aH
        return result

    def deindexify_state(self, num):
        num, prev_aH = (num // self.apomdp.nAH), (num % self.apomdp.nAH)
        num, next_aH = (num // self.apomdp.nAH), (num % self.apomdp.nAH)
        theta, s = (num // self.apomdp.nS), (num % self.apomdp.nS)
        return s, next_aH, prev_aH, theta

    def numpy_initial_state_distribution(self):
        dist = np.zeros(self.nS)
        dist_real = self.initial_state_distribution
        for s_tuple in dist_real.support():
            prob = self.initial_state_distribution.get_probability(s_tuple)
            s_real, next_aH_real, prev_aH_real, theta_real = s_tuple
            s = self.apomdp.state_to_index(s_real)
            next_aH = self.apomdp.human_action_to_index(next_aH_real)
            prev_aH = self.apomdp.human_action_to_index(prev_aH_real)
            theta = self.apomdp.theta_map[theta_real]
            state_num = self.indexify_state(s, next_aH, prev_aH, theta)
            dist[state_num] = prob
        return dist

    def indexify_obs(self, oR, prev_aH):
        return oR * self.apomdp.nAH + prev_aH

    def deindexify_obs(self, num):
        return (num // self.apomdp.nAH), (num % self.apomdp.nAH)

    def get_num_states(self):
        return self.nS

    def get_num_actions(self):
        return self.nA

    def get_transition_matrix(self):
        return self._T

    def get_reward_matrix(self):
        return self._R

    def reset(self):
        obs_encoded = super().reset()
        oR_real, prev_aH_real = self.apomdp.decode_obs(obs_encoded)
        oR = self.apomdp.robot_obs_to_index(oR_real)
        prev_aH = self.apomdp.human_action_to_index(prev_aH_real)
        obs = self.indexify_obs(oR, prev_aH)
        return obs

    def step(self, action):
        action_real = self.apomdp.index_to_robot_action(action)
        obs_encoded, reward, done, info =  super().step(action_real)
        oR_real, prev_aH_real = self.apomdp.decode_obs(obs_encoded)
        oR = self.apomdp.robot_obs_to_index(oR_real)
        prev_aH = self.apomdp.human_action_to_index(prev_aH_real)
        obs = self.indexify_obs(oR, prev_aH)
        return obs, reward, done, info

    def render(self, mode='human', prev_action=None):
        if prev_action != None:
            prev_action = self.apomdp.index_to_robot_action(prev_action)
        super().render(mode=mode, prev_action=prev_action)


class ReducedFullyObservableAssistancePOMDPWithMatrices(ReducedAssistancePOMDPWithMatrices):
    """When the underlying APOMDP is fully observable, the only uncertainty the
    robot has is in what the human will do (which depends on the unknown
    theta). So, we can make it so that the observation space only includes the
    previous human action, making solvers much more efficient.
    """
    def __init__(self, apomdp):
        super().__init__(apomdp)
        self.nO = self.apomdp.nAH
        obs_matrix = np.zeros((len(self.apomdp.thetas), self.apomdp.nS, self.apomdp.nAH, self.apomdp.nAH, self.nO))
        for o in range(self.nO):
            obs_matrix[:,:,:,o,o] = 1.0
        self._O = obs_matrix.reshape((self.nS, self.nO))
        self.sensor_model = TabularForwardSensorModel(self, self._O)

    def reset(self):
        obs = super().reset()
        oR, prev_aH = self.deindexify_obs(obs)
        return prev_aH

    def step(self, action):
        obs, reward, done, info = super().step(action)
        oR, prev_aH = self.deindexify_obs(obs)
        return prev_aH, reward, done, info


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
        yield self.x

    def get_probability(self, x):
        return 1.0 if x == self.x else 0.0

    def sample(self):
        return self.x


class DictionaryDistribution(Distribution):
    def __init__(self, x_to_prob_dict):
        self.x_to_prob_dict = x_to_prob_dict

    def support(self):
        for x in self.x_to_prob_dict.keys():
            yield x

    def get_probability(self, x):
        return self.x_to_prob_dict.get(x, 0.0)


class DiscreteDistribution(Distribution):
    def __init__(self, option_prob_map):
        if type(option_prob_map) == np.ndarray:
            assert len(option_prob_map.shape) == 1
            option_prob_map = dict(zip(range(len(option_prob_map)), option_prob_map))
        self.option_prob_map = option_prob_map

    def support(self):
        for x, p in self.option_prob_map.items():
            if p > 0:
                yield x

    def get_probability(self, option):
        return self.option_prob_map.get(option, 0.0)

    def sample(self):
        options, probs = zip(*self.option_prob_map.items())
        idx = np.random.choice(len(options), p=probs)
        return options[idx]


class UniformDiscreteDistribution(DiscreteDistribution):
    def __init__(self, options):
        p = 1.0 / len(options)
        super().__init__({ option:p for option in options })


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
        if prev_aH != self.prev_aH or theta != self.theta:
            return 0.0
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
        if prev_aH != self.apomdp.default_aH:
            return 0.0
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


def value_iteration(T, R, discount=0.9, num_iters=30, **kwargs):
    nS, nA, _ = T.shape
    Q = np.empty((nS, nA))
    V = np.zeros((nS,))

    for _ in range(num_iters):
        Q = (T*R).sum(axis=-1) + np.tensordot(T, discount * V, axes=(2, 0))
        V = np.max(Q, axis=1)

    policy = np.eye(nA)[Q.argmax(axis=1)]
    return policy
