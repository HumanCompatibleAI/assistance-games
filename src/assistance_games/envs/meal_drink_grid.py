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
import assistance_games.rendering as rendering
from assistance_games.utils import get_asset, sample_distribution, dict_to_sparse

class MealDrinkGridAG(AssistanceGame):
    State = namedtuple('State', ['pos_x', 'pos_y', 'meal', 'meal_timer', 'drink', 'h_away_timer', 'query'])
    def __init__(self, spec):
        self.height = spec.height
        self.width = spec.width
        self.meal_pos = spec.meal_pos
        self.drink_pos = spec.drink_pos
        self.horizon = spec.horizon
        self.meal_cooking_time = spec.meal_cooking_time
        self.max_h_away_timer = spec.max_h_away_timer
        n_world_actions = 7
        self.n_queries = 2
        # robot actions are noop, 4 actions for moving, 2 for cooking, and two queries
        self.robot_action_space = Discrete(n_world_actions + self.n_queries)
        # human actions are no-op and the binary query response (separate queries don't get separate response actions)
        self.human_action_space = Discrete(3)

        self.enumerate_states()

        init_state = self.State(pos_y=0, pos_x=0, meal=0, meal_timer=0, drink=0, h_away_timer=spec.init_h_away_timer, query=0)
        init_state_dist = np.zeros(self.nS)
        init_state_dist[self.get_idx(init_state)] = 1.0
        transition, rewards_dist = self.make_transition_and_reward_matrices()

        # How many different values can each state feature take. This shouldn't be interpreted as a state!
        self.discrete_feature_dims = self.State(pos_y=self.height,
                                                pos_x=self.width,
                                                meal=5,
                                                meal_timer=spec.meal_cooking_time + 1,
                                                drink=4,
                                                h_away_timer=self.max_h_away_timer + 1,
                                                query=self.n_queries + 1)

        self.one_hot_features = {'pos_x', 'pos_y', 'meal', 'meal_timer', 'drink', 'h_away_timer', 'query'}
        num_regular_features = len(init_state) - len(self.one_hot_features)
        len_one_hot_features = 0
        for feature in self.one_hot_features:
            assert hasattr(init_state, feature)
            len_one_hot_features += getattr(self.discrete_feature_dims, feature)
        self.feature_vector_length = num_regular_features + len_one_hot_features
        self.feature_matrix = self.make_feature_matrix()
        self.max_feature_value = np.max(self.feature_matrix)

        super().__init__(
            state_space=Discrete(self.nS),
            human_action_space=self.human_action_space,
            robot_action_space=self.robot_action_space,
            transition=transition,
            reward_distribution=rewards_dist,
            initial_state_distribution=init_state_dist,
            horizon=self.horizon,
            discount=spec.discount,
        )

    def transition_fn(self, s, a_h, a_r):
        # action-independent transitions
        # if current timestep is the env's horizon, transition w/o increasing the state timestep
        # if currently asking a question, transition to the same state but w/o question
        if s.query > 0:
            return self.State(pos_x=s.pos_x,
                              pos_y=s.pos_y,
                              meal=s.meal,
                              meal_timer=s.meal_timer,
                              drink=s.drink,
                              h_away_timer=max(s.h_away_timer - 1, 0),
                              query=0)

        new_pos_x, new_pos_y, new_meal, new_meal_timer, new_drink, new_query = s.pos_x, s.pos_y, s.meal, s.meal_timer, s.drink, s.query

        # If the meal or the drink is ready, it is automatically served / transitioned into an own absorbing state.
        # This way the agent spends only one timestep in a state with pizza / drink, and is rewarded once.
        if s.meal in [2, 3] and s.meal_timer == 0:
            new_meal = 4
        if s.drink in [1, 2]:
            new_drink = 3
        # tick down the baking timer if it's not 0
        if s.meal_timer > 0 and s.meal in [2, 3]:
            new_meal_timer = s.meal_timer - 1

        # movement
        if a_r in [1, 2, 3, 4]:
            move = [(1, 0), (-1, 0), (0, -1), (0, 1)][a_r - 1]
            new_pos = (s.pos_y + move[0], s.pos_x + move[1])
            if (0 <= new_pos[0] <= self.height - 1 and 0 <= new_pos[1] <= self.width - 1
                    and new_pos not in [self.meal_pos, self.drink_pos]):
                new_pos_y, new_pos_x = new_pos[0], new_pos[1]

        # cooking
        elif a_r in [5, 6]:
            # meal
            if (s.pos_y, s.pos_x + 1) == self.meal_pos:
                # make dough
                if s.meal == 0:
                    new_meal = 1
                # bake pizza or cake using dough; start the timer for cooking_time_needed
                elif s.meal == 1:
                    new_meal_timer = self.meal_cooking_time
                    if a_r == 5:
                        new_meal = 2
                    elif a_r == 6:
                        new_meal = 3
                # elif s.meal in [2, 3]:
                #     # tick down the baking timer if it's not 0
                #     if s.meal_timer > 0:
                #         new_meal_timer -= 1

            # drink
            elif (s.pos_y, s.pos_x + 1) == self.drink_pos:
                if s.drink == 0:
                    if a_r == 5:
                        new_drink = 1
                    elif a_r == 6:
                        new_drink = 2

        # asking one of the two queries
        elif a_r in [7, 8]:
            new_query = 1 if a_r == 7 else 2

        return self.State(pos_x=new_pos_x,
                          pos_y=new_pos_y,
                          meal=new_meal,
                          meal_timer=new_meal_timer,
                          drink=new_drink,
                          h_away_timer=max(s.h_away_timer - 1, 0),
                          query=new_query)

    def enumerate_states(self):
        state_num = {}
        for x in range(self.width):
            for y in range(self.height):
                # can't be at the same position as a meal or a drink
                if (y, x) in [self.meal_pos, self.drink_pos]:
                    continue
                for meal in range(5):
                    for meal_timer in range(self.meal_cooking_time + 1):
                        # timer can only be active for the cooking pizza / cake states
                        if meal_timer > 0 and meal not in [2, 3]:
                            continue
                        for drink in range(4):
                            for h_away_timer in range(self.max_h_away_timer + 1):
                                for query in range(self.n_queries + 1):
                                    s = self.State(pos_x=x,
                                                   pos_y=y,
                                                   meal=meal,
                                                   meal_timer=meal_timer,
                                                   drink=drink,
                                                   h_away_timer=h_away_timer,
                                                   query=query)
                                    state_num[s] = len(state_num)
        self.state_num = state_num
        self.num_state = {v: k for k, v in self.state_num.items()}
        self.nS = len(state_num)
        print('The number of states is ', self.nS)

    def get_idx(self, state):
        return self.state_num[state]

    def get_state(self, idx):
        return self.num_state[idx]

    def make_transition_and_reward_matrices(self):
        T_dict = {}
        reward_dicts = {(True, True): {}, (True, False): {}, (False, True): {}, (False, False):{}}

        T_shape = (self.nS, self.human_action_space.n, self.robot_action_space.n, self.nS)
        for idx in range(self.nS):
            s = self.get_state(idx)
            for a_r in range(self.robot_action_space.n):
                next_idx = self.get_idx(self.transition_fn(s=s, a_h=0, a_r=a_r))
                for a_h in range(self.human_action_space.n):
                    T_dict[idx, a_h, a_r, next_idx] = 1.0
                    # print('TRANSITION: ', s, a_r, a_h, self.get_state(next_idx))
                    # rewards
                    for prefer_cake in [True, False]:
                        for prefer_lemonade in [True, False]:
                            reward_dicts[(prefer_cake, prefer_lemonade)][idx, a_h, a_r, next_idx] \
                                = self.reward_fn(s, prefer_cake, prefer_lemonade)
        transitions = dict_to_sparse(T_dict, T_shape)
        rewards_dist = [(dict_to_sparse(R, T_shape), 0.25) for R in reward_dicts.values()]
        return transitions, rewards_dist

    def reward_fn(self, s, prefer_pizza=False, prefer_lemonade=False):
        r = 0
        if s.meal_timer == 0 and ((s.meal == 2 and prefer_pizza) or (s.meal == 3 and not prefer_pizza)):
            r += 2
        elif s.meal_timer == 0 and ((s.meal == 2 and not prefer_pizza) or (s.meal == 3 and prefer_pizza)):
            r += -1

        if (s.drink == 1 and prefer_lemonade) or (s.drink == 2 and not prefer_lemonade):
            r += 2
        elif (s.drink == 1 and not prefer_lemonade) or (s.drink == 2 and prefer_lemonade):
            r += -1
        return r if s.query == 0 else 0

    def make_feature_matrix(self):
        feature_matrix = np.zeros((self.nS, self.feature_vector_length))
        for s_idx in range(self.nS):
            feature_matrix[s_idx, :] = self.feature_function(self.get_state(s_idx))
        return feature_matrix

    def feature_function(self, s):
        # s is the flat namedtuple with attributes ['pos_x', 'pos_y', 'meal', 'meal_timer', 'drink', 'query', 'time']
        f_vec = np.zeros(self.feature_vector_length, dtype='float32')
        i = 0
        for feature in s._fields:
            if feature in self.one_hot_features:
                f_vec[i + getattr(s, feature)] = 1
                i += getattr(self.discrete_feature_dims, feature)
            else:
                f_vec[i] = getattr(s, feature)
                i += 1
        return f_vec

    def get_state_features(self, s_idx):
        return self.feature_matrix[s_idx, :]


def human_response_meal_drink_grid(assistance_game, reward, **kwargs):
    ag = assistance_game
    reward_idx = kwargs['reward_idx']
    prefer_pizza, prefer_lemonade = [(True, True), (True, False), (False, True), (False, False)][reward_idx]
    policy = np.zeros((ag.state_space.n, ag.human_action_space.n))
    for idx in range(ag.nS):
        s = ag.get_state(idx)
        # noop
        if s.query == 0 or s.h_away_timer > 0:
            policy[idx, 0] = 1.0
        # meal query
        elif s.query == 1:
            if prefer_pizza:
                policy[idx, 1] = 1.0
            else:
                policy[idx, 2] = 1.0
        # drink query
        elif s.query == 2:
            if prefer_lemonade:
                policy[idx, 1] = 1.0
            else:
                policy[idx, 2] = 1.0
    return policy


class MealDrinkGridProblem(AssistanceProblem):
    Spec = namedtuple('Spec', ['height',
                               'width',
                               'meal_pos',
                               'meal_cooking_time',
                               'drink_pos',
                               'horizon',
                               'discount',
                               'init_h_away_timer',
                               'max_h_away_timer'])
    def __init__(self, use_belief_space=True):
        spec = self.Spec(height=5,
                         width=5,
                         meal_pos=(4, 4),
                         meal_cooking_time=4,
                         drink_pos=(0, 4),
                         horizon=30,
                         discount=0.99,
                         init_h_away_timer=3,
                         max_h_away_timer=3)
        human_policy_fn = human_response_meal_drink_grid
        self.assistance_game = MealDrinkGridAG(spec)
        ag = self.assistance_game

        if use_belief_space:
            observation_model_fn = BeliefObservationModel
        else:
            feature_extractor = lambda s_idx: ag.get_state_features(s_idx % ag.state_space.n)
            setattr(feature_extractor, 'n', ag.feature_vector_length)
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
        s_str = 'pos: ({}, {}), meal: {}, meal_timer: {}, drink: {}, h_away_timer: {}'.format(s.pos_y,
                                                                                   s.pos_x,
                                                                                   s.meal,
                                                                                   s.meal_timer,
                                                                                   s.drink,
                                                                                   s.h_away_timer)
        if s.query:
            s_str = s_str + ' query={}'.format(s.query)
        print(s_str)




















