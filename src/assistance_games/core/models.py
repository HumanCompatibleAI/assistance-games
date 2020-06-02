import numpy as np
from gym.spaces import Discrete, MultiDiscrete, Box
import sparse

from assistance_games.utils import sample_distribution


### Spaces

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

class FunctionalTransitionModel(TransitionModel):
    def __init__(self, pomdp, fn):
        super().__init__(pomdp)
        self.fn = fn

    def __call__(self):
        return self.fn(self.pomdp.state, self.pomdp.action)


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

class DiscreteFeatureSenseObservationModel(ObservationModel):
    def __init__(self, pomdp, feature_extractor):
        super().__init__(pomdp)
        self.feature_extractor = feature_extractor

    def __call__(self):
        feature = self.feature_extractor(self.pomdp.state)

        sense = self.pomdp.sensor_model.sense
        if sense is None:
            sense = self.pomdp.sensor_model.space.sample()

        return np.array([feature, sense])

    @property
    def space(self):
        num_senses = self.pomdp.sensor_model.space.n
        num_features = self.feature_extractor.n
        return MultiDiscrete([num_features, num_senses])

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
        return obs

    @property
    def space(self):
        num_senses = self.pomdp.sensor_model.space.n
        num_features = self.feature_extractor.n
        return Box(low=0.0, high=self.pomdp.assistance_game.max_feature_value, shape=(num_features + num_senses, 1))


class FunctionalObservationModel(ObservationModel):
    def __init__(self, pomdp, fn, space):
        super().__init__(pomdp)
        self.fn = fn
        self.space = space

    def __call__(self):
        state = self.pomdp.state
        sense = self.pomdp.sensor_model.sense
        return self.fn(state=state, sense=sense)

### Sensor models

class SensorModel:
    def __init__(self, pomdp):
        self.pomdp = pomdp
        self.sense = None

    def __call__(self):
        pass

class TabularForwardSensorModel(SensorModel):
    def __init__(self, pomdp, sensor):
        super().__init__(pomdp)
        self.sensor = sensor

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
        super().__init__(pomdp)
        self.back_sensor = back_sensor

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

class FunctionalRewardModel(RewardModel):
    def __init__(self, pomdp, fn):
        super().__init__(pomdp)
        self.fn = fn

    def __call__(self):
        prev_state = self.pomdp.prev_state
        action = self.pomdp.action
        state = self.pomdp.state
        return self.fn(prev_state, action, state)

class ShapedFunctionalRewardModel(RewardModel):
    def __init__(self, pomdp, fn, shaping_fns):
        super().__init__(pomdp)
        self.fn = fn
        self.shaping_fns = shaping_fns

    def __call__(self):
        s1 = self.pomdp.prev_state
        action = self.pomdp.action
        s2 = self.pomdp.state
        gamma = self.pomdp.discount
        done = self.pomdp.done
        base_reward = self.fn(s1, action, s2)
        # Shaping rewards: add gammma phi(s2) - phi(s1) for each shaping function phi
        # If done is true, imagine that we undergo one more transition to an absorbing state where all the phis are zero
        # Then we add gamma phi(s2) - phi(s1) + gamma * (0 - phi(s2)) = - phi(s)
        # Proper reward shaping should have "if done:" below, we use "if False:" to turn off the last step reward shaping
        # This effectively allows the agent to "keep" any shaping reward it has
        # on the last step, which drastically increases the effectiveness of
        # shaping, but also loses the guarantee of leaving the optimal policy unchanged.
        if False:
            shaping_reward = -sum([fn(s1) for fn in self.shaping_fns])
        else:
            shaping_reward = sum([gamma * fn(s2) - fn(s1) for fn in self.shaping_fns])
        return base_reward + shaping_reward
            


### Termination models

class TerminationModel:
    def __init__(self, pomdp):
        self.pomdp = pomdp

    def __call__(self):
        pass


class NoTerminationModel(TerminationModel):
    def __call__(self):
        return False
