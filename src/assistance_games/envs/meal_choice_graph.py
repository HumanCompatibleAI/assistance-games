"""Ready-to-use POMDP and AssistanceProblem environments.
"""
from collections import namedtuple
from copy import copy, deepcopy
from itertools import product

import os

import gym
from gym.spaces import Discrete, MultiDiscrete
import numpy as np
import pkg_resources
import sparse

from functools import partial

from assistance_games.core import (
    POMDP,
    AssistanceGame,
    AssistanceProblem,
    get_human_policy,
    BeliefObservationModel,
    FeatureSenseObservationModel,
    discrete_reward_model_fn_builder,
)

from assistance_games.parser import read_pomdp
from assistance_games.utils import get_asset, sample_distribution, dict_to_sparse

class MealChoiceTimeDependentAG(AssistanceGame):
    State = namedtuple('State', ['s_w', 'query', 'time'])

    def __init__(self, horizon=20):
        self.horizon = horizon
        self.max_feature_value = horizon
        n_world_states = 5
        self.n_world_states = n_world_states
        n_world_actions = 3
        self.num_world_actions = n_world_actions
        n_queries = 1
        self.n_queries = n_queries

        state_space = Discrete((n_world_states + n_world_states * n_queries) * horizon + 1)

        self.state_space = state_space
        human_action_space = Discrete(3)
        robot_action_space = Discrete(n_world_actions + n_queries)

        reward0 = np.zeros((state_space.n, human_action_space.n, robot_action_space.n, state_space.n))
        reward1 = np.zeros_like(reward0)
        transition = np.zeros((state_space.n, human_action_space.n, robot_action_space.n, state_space.n))
        for s_idx in range(state_space.n):
            s = self.get_state(s_idx)
            assert s_idx == self.get_idx(s)
            for a_r in range(robot_action_space.n):
                # there is no loop over the human actions as they don't affect the resulting state
                transition[s_idx, :, a_r, self.transition_state_id(s_idx, a_r)] = 1
                reward0[s_idx, :, a_r, :] = self.reward_fn(s, a_r, world_rewards=[0, 0., 2., -1., 0])
                reward1[s_idx, :, a_r, :] = self.reward_fn(s, a_r, world_rewards=[0, 0., -1., 2., 0])

        initial_state_dist = np.zeros(state_space.n)
        initial_state_dist[0] = 1.0

        super().__init__(
            state_space=state_space,
            human_action_space=human_action_space,
            robot_action_space=robot_action_space,
            transition=transition,
            reward_distribution=[(reward0, 0.5), (reward1, 0.5)],
            initial_state_distribution=initial_state_dist,
            horizon=horizon,
            discount=0.9,
        )

    def get_state(self, s_idx):
        time = s_idx // (self.n_world_states * (self.n_queries + 1))
        s_idx = s_idx % (self.n_world_states * (self.n_queries + 1))
        s_w = s_idx % self.n_world_states
        query = s_idx // self.n_world_states
        return self.State(s_w=s_w, query=query, time=time)

    def get_idx(self, s):
        return s.s_w + self.n_world_states * s.query + (self.n_world_states * (self.n_queries + 1)) * s.time

    def reward_fn(self, s, a_r, world_rewards):
        # Being in the querying state or doing the no-op or the query action brings no reward.
        # Because all other world actions change the state, this ensures that the reward is collected only once
        return world_rewards[s.s_w] if (s.query == 0 and 0 < a_r < self.num_world_actions) else 0

    def transition_state(self, s, robot_action):
        # if the robot is currently waiting for human's response, transition back to the world state
        # (human action doesn't affect this at all)
        if s.query > 0:
            return self.State(s_w=s.s_w,
                              query=0,
                              time=min(s.time + 1, self.horizon - 1))
        # if the robot is asking a question, transition to the corresponding query state
        if robot_action > self.num_world_actions-1:
            return self.State(s_w=s.s_w,
                              query=robot_action - self.num_world_actions + 1,
                              time=min(s.time + 1, self.horizon - 1))
        # else do a world state transition
        else:
            return self.State(s_w=self.transition_world(s.s_w, robot_action),
                              query=0,
                              time=min(s.time + 1, self.horizon - 1))

    def transition_world(self, world_state, robot_action):
        # transitions for the toy 5-state graph mdp
        s_w, a_r = world_state, robot_action
        assert type(s_w) is int
        # the query and the no-op actions don't change the world state
        if a_r >= self.num_world_actions or a_r == 0: return s_w

        if s_w == 0: return 1
        elif s_w == 1:
            if a_r == 1: return 2
            elif a_r == 2: return 3
        elif s_w in [2, 3]: return 4
        elif s_w == 4: return 4

    def transition_state_id(self, s_idx, robot_action):
        """Given the current state id and the robot action, outputs the id of the next state"""
        s = self.get_state(s_idx)
        return self.get_idx(self.transition_state(s, robot_action))

    def get_state_features(self, s):
        # s is the flat namedtuple state
        features = np.zeros(self.n_world_states + 2, dtype='float32')
        features[s.s_w] = 1
        features[-1] = s.time
        features[-2] = s.query
        return features
    

def query_response_meal_choice_time_dep(time_before_feedback_available=0):
    def time_dep_policy_fn(assistance_game, reward, **kwargs):
        """Hardcoded query response for the time-dependent meal choice game. If in the querying state and able to give
        feedback, the human performs action 1 if she prefers cake and action 2 if she prefers pizza. Otherwise the human
        does action 0, corresponding to no-op"""
        ag = assistance_game
        policy = np.zeros((ag.state_space.n, ag.human_action_space.n))

        for s_idx in range(ag.state_space.n):
            s = ag.get_state(s_idx)
            if s.query > 0 and s.time >= time_before_feedback_available:
                if reward[2, 0, 1, 0] > reward[3, 0, 1, 0]:
                    policy[s_idx, 1] = 1
                else:
                    policy[s_idx, 2] = 1
            else:
                # no-op
                policy[s_idx, 0] = 1
        return policy
    return time_dep_policy_fn


class MealChoiceTimeDependentProblem(AssistanceProblem):
    def __init__(self, use_belief_space=True):
        human_policy_fn = query_response_meal_choice_time_dep(time_before_feedback_available=4)
        self.assistance_game = MealChoiceTimeDependentAG(horizon=10)
        ag = self.assistance_game
        if use_belief_space:
            observation_model_fn = BeliefObservationModel
        else:
            # feature_extractor = lambda state : state % self.assistance_game.state_space.n
            feature_extractor = lambda s_idx: ag.get_state_features(ag.get_state(s_idx % ag.state_space.n))
            init_s_idx = np.where(ag.initial_state_distribution)[0][0]
            setattr(feature_extractor, 'n', len(ag.get_state_features(ag.get_state(init_s_idx))))
            observation_model_fn = partial(FeatureSenseObservationModel, feature_extractor=feature_extractor)

        reward_model_fn_builder = partial(discrete_reward_model_fn_builder, use_belief_space=use_belief_space)
        super().__init__(
            assistance_game=self.assistance_game,
            human_policy_fn=human_policy_fn,
            observation_model_fn=observation_model_fn,
            reward_model_fn_builder=reward_model_fn_builder,
        )

    def render(self, mode='human'):
        s = self.assistance_game.get_state(self.state % self.assistance_game.state_space.n)
        wanted_meal = ['cake', 'pizza'][self.state // self.assistance_game.state_space.n]
        s_w_str = ['flour', 'dough', 'cake', 'pizza', 'absorbing']
        s_query = ' query = {}'.format(s.query) if s.query else ''
        print('s = {}, t = {}, human wants {}'.format(s_w_str[s.s_w],  s.time, wanted_meal) + s_query)




