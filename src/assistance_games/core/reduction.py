import numpy as np

from assistance_games.core.pomdp import POMDP
from assistance_games.core.distributions import Distribution
from assistance_games.utils import dict_to_sparse


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

    def get_obs_distribution(self, state):
        s, next_aH, prev_aH, theta = state
        base_dist = self.apomdp.get_robot_obs_distribution(s)
        return self.apomdp.encode_obs_distribution(base_dist, prev_aH)

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
                obsR_real_dist = self.apomdp.get_robot_obs_distribution(s_real)
                for obsR_real in obsR_real_dist.support():
                    
                    obsR = self.apomdp.robot_obs_to_index(obsR_real)
                    new_obs = self.indexify_obs(obsR, prev_aH)
                    O_dict[new_s, new_obs] = obsR_real_dist.get_probability(obsR_real)

                    obsH_real_dist = self.apomdp.get_human_obs_distribution(s2_real)
                    for obsH_real in obsH_real_dist.support():
                        p_obsH = obsH_real_dist.get_probability(obsH_real)
                        policy = self.apomdp.get_human_action_distribution(obsH_real, aR_real, theta_real)
                        for next_aH_real in policy.support():
                            p_next_aH = policy.get_probability(next_aH_real)
                            next_aH = self.apomdp.human_action_to_index(next_aH_real)

                            new_s2 = self.indexify_state(s2, next_aH, aH, theta)
                            key = (new_s, aR, new_s2)
                            T_dict[key] = T_dict.get(key, 0) + p * p_next_aH * p_obsH
                            R_dict[key] = r

        self._T = dict_to_sparse(T_dict, (self.nS, self.nA, self.nS))
        self._R = dict_to_sparse(R_dict, (self.nS, self.nA, self.nS))
        self._O = dict_to_sparse(O_dict, (self.nS, self.nO))

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

    def get_observation_matrix(self):
        return self._O

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

    def reset(self):
        obs = super().reset()
        oR, prev_aH = self.deindexify_obs(obs)
        return prev_aH

    def step(self, action):
        obs, reward, done, info = super().step(action)
        oR, prev_aH = self.deindexify_obs(obs)
        return prev_aH, reward, done, info


class ExtendedTransitionDistribution(Distribution):
    def __init__(self, underlying_transition_distribution, prev_aH, prev_aR, theta, apomdp):
        self.trans_dist = underlying_transition_distribution
        self.prev_aH = prev_aH
        self.prev_aR = prev_aR
        self.theta = theta
        self.apomdp = apomdp

    def support(self):
        yielded = []
        for next_state in self.trans_dist.support():
            obsH_dist = self.apomdp.get_human_obs_distribution(next_state)
            policy = self.apomdp.get_human_action_distribution(obsH, self.prev_aR, self.theta)
            for next_aH in policy.support():
                key = (next_state, next_aH, self.prev_aH, self.theta)
                if key not in yielded:
                    yielded.append(key)
                    yield key

    def get_probability(self, x):
        next_state, next_aH, prev_aH, theta = x
        if prev_aH != self.prev_aH or theta != self.theta:
            return 0.0

        prob_next_aH = 0.0
        obsH_dist = self.apomdp.get_human_obs_distribution(next_state)
        for obsH in obsH_dist.support():
            policy = self.apomdp.get_human_action_distribution(obsH, self.prev_aR, self.theta)
            prob_next_aH += obsH_dist.get_probability(obsH) * policy.get_probability(next_aH)
        return self.trans_dist.get_probability(next_state) * prob_next_aH

    def sample(self):
        next_state = self.trans_dist.sample()
        obsH = self.apomdp.get_human_obs_distribution(next_state).sample()
        next_aH = self.apomdp.get_human_action_distribution(obsH, self.prev_aR, self.theta).sample()
        return next_state, next_aH, self.prev_aH, self.theta


class ExtendedInitialStateDistribution(Distribution):
    def __init__(self, apomdp):
        self.apomdp = apomdp

    def support(self):
        yielded = []
        prev_aH, prev_aR = self.apomdp.default_aH, self.apomdp.default_aR
        for state in self.apomdp.initial_state_distribution.support():
            for theta in self.apomdp.theta_distribution.support():
                for obsH in self.apomdp.get_human_obs_distribution(state).support():
                    for next_aH in self.apomdp.get_human_action_distribution(obsH, prev_aR, theta).support():
                        key = state, next_aH, prev_aH, theta
                        if key not in yielded:
                            yielded.append(key)
                            yield key

    def get_probability(self, x):
        state, next_aH, prev_aH, theta = x
        if prev_aH != self.apomdp.default_aH:
            return 0.0
        prob_next_aH = 0.0
        obsH_dist = self.apomdp.get_human_obs_distribution(state)
        for obsH in obsH_dist.support():
            policy = self.apomdp.get_human_action_distribution(obsH, self.apomdp.default_aR, theta)
            prob_next_aH += obsH_dist.get_probability(obsH) * policy.get_probability(next_aH)

        result = prob_next_aH
        result *= self.apomdp.theta_distribution.get_probability(theta)
        result *= self.apomdp.initial_state_distribution.get_probability(state)
        return result

    def sample(self):
        state = self.apomdp.initial_state_distribution.sample()
        theta = self.apomdp.theta_distribution.sample()
        obsH = self.apomdp.get_human_obs_distribution(state).sample()
        next_aH = self.apomdp.get_human_action_distribution(obsH, self.apomdp.default_aR, theta).sample()
        prev_aH = self.apomdp.default_aH
        return state, next_aH, prev_aH, theta
