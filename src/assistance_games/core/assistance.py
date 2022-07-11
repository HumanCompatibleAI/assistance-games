from abc import ABC, abstractmethod
import numpy as np

from assistance_games.core.distributions import KroneckerDistribution, DiscreteDistribution
from assistance_games.utils import dict_to_sparse


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
    
    def get_human_obs_distribution(self, state):
        # Assume full observability by default
        return KroneckerDistribution(state)

    def get_robot_obs_distribution(self, state):
        # Assume full observability by default
        return KroneckerDistribution(state)

    @abstractmethod
    def encode_obs_distribution(self, obs_dist, prev_aH):
        """Once converted to a POMDP, observations are tuples (obs,
        prev_aH). This method allows postprocessing, e.g. to make it suitable
        for input to a policy net.
        """
        pass

    @abstractmethod
    def get_transition_distribution(self, state, aH, aR):
        """Returns the distribution T( . | state, aH, aR)."""
        pass

    @abstractmethod
    def get_reward(self, state, aH, aR, next_state, theta):
        """Returns the reward for transition (state, aH, aR, next_state)."""
        pass

    @abstractmethod
    def get_human_action_distribution(self, obsH, aR, theta):
        """Returns the human policy PiH( . | obsH, aR, theta)."""
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

    def get_prob_aH(self, aH, state, aR, theta):
        prob = 0.0
        obsH_dist = self.get_human_obs_distribution(state)
        for obsH in obsH_dist.support():
            policy = self.get_human_action_distribution(obsH, aR, theta)
            prob += obsH_dist.get_probability(obsH) * policy.get_probability(aH)
        return prob


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

    def get_human_action_distribution(self, obsH, aR, theta):
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



def value_iteration(T, R, discount=0.9, num_iters=30, **kwargs):
    nS, nA, _ = T.shape
    Q = np.empty((nS, nA))
    V = np.zeros((nS,))

    for _ in range(num_iters):
        Q = (T*R).sum(axis=-1) + np.tensordot(T, discount * V, axes=(2, 0))
        V = np.max(Q, axis=1)

    policy = np.eye(nA)[Q.argmax(axis=1)]
    return policy
