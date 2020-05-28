"""Core classes, such as POMDP and AssistanceGame.
"""

import functools

import numpy as np
import gym
from gym.spaces import Discrete, MultiDiscrete, Box
import sparse
from scipy.special import logsumexp

from assistance_games.utils import sample_distribution, uniform_simplex_sample, force_sparse

from assistance_games.core.builders import (
    back_sensor_model_fn_builder,
    discrete_reward_model_fn_builder,
    discrete_state_space_builder,
    tabular_transition_model_fn_builder,
)

from assistance_games.core.models import (
    BeliefObservationModel,
    NoTerminationModel,
)


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


class AssistanceProblem(POMDP):
    def __init__(
        self,
        assistance_game,
        human_policy_fn,
        state_space_builder=discrete_state_space_builder,
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
        *builders : AssistanceGame -> Reward -> Model
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
