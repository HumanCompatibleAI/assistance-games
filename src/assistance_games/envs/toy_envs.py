"""Ready-to-use POMDP and AssistanceProblem environments.
"""
from collections import namedtuple, Counter
from copy import copy, deepcopy
from itertools import product

import os
import itertools

import gym
from gym.spaces import Discrete, MultiDiscrete, Box
import numpy as np
import pkg_resources
import sparse

from functools import partial

from assistance_games.core import (
    POMDP,
    AssistanceGame,
    AssistanceProblem,
    get_human_policy,
    functional_random_policy_fn,
    BeliefObservationModel,
    FunctionalObservationModel,
    FunctionalTransitionModel,
    FunctionalRewardModel,
    SensorModel,
    DiscreteFeatureSenseObservationModel,
    discrete_reward_model_fn_builder,
)

from assistance_games.parser import read_pomdp
import assistance_games.rendering as rendering
from assistance_games.utils import get_asset, sample_distribution, dict_to_sparse

# ***** POMDPs ******

class TwoStatePOMDP(POMDP):
    """Russell and Norvig's two-state POMDP.

    There is a reward of +1 for staying in the second state,
    but the agents' observations are noisy, so they start unsure
    of where they are.

    There is a 0.3 chance of getting the wrong state as observation,
    and a 0.1 chance of moving to the wrong state.
    """
    def __init__(self):
        self.state_space = Discrete(2)
        self.action_space = Discrete(2)
        self.sensor_space = Discrete(2)

        noise = 0.3

        self.sensor = np.array(
            [[1.0 - noise, noise      ],
             [noise      , 1.0 - noise]]
        )

        act_noise = 0.1

        self.transition = np.zeros((2, 2, 2))
        self.transition[:, 0, :] = (1 - act_noise) * np.eye(2) + act_noise * np.eye(2)[[1, 0]]
        self.transition[:, 1, :] = act_noise * np.eye(2) + (1 - act_noise) * np.eye(2)[[1, 0]]

        self.horizon = 4

        self.rewards = np.array([0, 1])
        
        self.initial_state = np.array([0.5, 0.5])
        self.initial_belief = np.array([0.5, 0.5])


class FourThreeMaze(POMDP):
    """Russell and Norvig's 4x3 maze

    The maze looks like this:

      ######
      #   +#
      # # -#
      #    #
      ######

    The + indicates a reward of 1.0, the - a penalty of -1.0.
    The # in the middle of the maze is an obstruction.
    Rewards and penalties are associated with states, not actions.
    The default reward/penalty is -0.04.

    States are numbered from left to right:

    0  1  2  3
    4     5  6
    7  8  9  10

    The actions, NSEW, have the expected result 80% of the time, and
    transition in a direction perpendicular to the intended on with a 10%
    probability for each direction.  Movement into a wall returns the agent
    to its original state.

    Observation is limited to two wall detectors that can detect when a
    a wall is to the left or right.  This gives the following possible
    observations:

    left, right, neither, both, good, bad, and absorb

    good = +1 reward, bad = -1 penalty, 
    """

    def __init__(self, *args, terminal=False, horizon=None, **kwargs):
        if terminal:
            pomdp = read_pomdp(get_asset('pomdps/four-three-terminal.pomdp'))
        else:
            pomdp = read_pomdp(get_asset('pomdps/four-three.pomdp'))
        self.__dict__ = pomdp.__dict__
        if horizon is not None:
            self.horizon = horizon
        self.viewer = None

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-120, 120, -120, 120)

            grid_side = 50
            self.grid = rendering.Grid(start=(-100, 100), grid_side=grid_side, shape=(4, 3), invert_y=True)
            self.viewer.add_geom(self.grid)

            agent_image = get_asset('images/robot1.png')
            agent = rendering.Image(agent_image, grid_side, grid_side)
            self.agent_transform = rendering.Transform()
            agent.add_attr(self.agent_transform)
            self.viewer.add_geom(agent)

            def coords_from_state(state):
                return self.grid.coords_from_state(state + int(state >= 5))
            self.coords_from_state = coords_from_state

            self.agent_transform.set_translation(*self.coords_from_state(self.state))

            self.viewer.add_geom(agent)

            hole_x, hole_y = self.grid.coords_from_state(5)
            l, r, t, b = -10, 10, 10, -10
            l, r = hole_x - grid_side / 2, hole_x + grid_side / 2, 
            t, b = hole_y - grid_side / 2, hole_y + grid_side / 2, 
            hole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.viewer.add_geom(hole)

        def add_bar(state, ratio):
            x, y = self.coords_from_state(state)
            xs, ys = self.grid.x_step, self.grid.y_step
            l, r = x - xs/2, x - xs/2 + 5
            b = y + ys/2
            t = b - ratio * ys
            bar = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            bar.set_color(0.7, 0.3, 0.3)
            self.viewer.add_onetime(bar)

        for state, ratio in enumerate(self.belief):
            add_bar(state, ratio)
        
        self.agent_transform.set_translation(*self.coords_from_state(self.state))

        self.viewer.render(return_rgb_array = mode=='rgb_array')

        return None

# ***** Assistance Problems ******


class RedBlueAssistanceGame(AssistanceGame):
    """

    This is the map of the assistance game:

    ..... .....
    .r b. .rRb.
    ..H.. .....
    .....

    H = Human
    R = Robot
    r = red ball
    b = blue ball

    The human prefers either the red or blue ball.
    The human and robot are separated, but both of them can
    either pick a red or a blue ball. Since the robot is
    unaware of which ball the human prefers, the optimal
    strategy for the robot is to wait for the human to pick.

    """
    def __init__(self):
        human_state_space = Discrete(4)
        robot_state_space = Discrete(3)

        human_action_space = Discrete(2)
        robot_action_space = Discrete(3)

        human_transition = np.zeros((human_state_space.n, human_action_space.n, human_state_space.n))
        human_transition[0, :, 1] = 1.0

        human_transition[1, 0, 2] = 1.0
        human_transition[1, 1, 3] = 1.0

        human_transition[2, :, 2] = 1.0
        human_transition[3, :, 3] = 1.0

        robot_transition = np.zeros((robot_state_space.n, robot_action_space.n, robot_state_space.n))
        robot_transition[0, 0, 1] = 1.0
        robot_transition[0, 1, 2] = 1.0
        robot_transition[0, 2, 0] = 1.0

        robot_transition[1, :, 1] = 1.0
        robot_transition[2, :, 2] = 1.0

        reward_0_h = np.zeros((human_state_space.n, human_action_space.n, human_state_space.n))
        reward_0_h[1, 0] = 1.0
        reward_0_r = np.zeros((robot_state_space.n, robot_action_space.n, robot_state_space.n))
        reward_0_r[0, 0] = 1.0

        reward_1_h = np.zeros((human_state_space.n, human_action_space.n, human_state_space.n))
        reward_1_h[1, 1] = 1.0
        reward_1_r = np.zeros((robot_state_space.n, robot_action_space.n, robot_state_space.n))
        reward_1_r[0, 1] = 1.0


        num_states = human_state_space.n * robot_state_space.n
        state_space = Discrete(num_states)

        def state_idx(human_state, robot_state):
            return robot_state_space.n * human_state + robot_state

        transition = np.zeros((state_space.n, human_action_space.n, robot_action_space.n, state_space.n))

        reward0 = np.zeros((state_space.n, human_action_space.n, robot_action_space.n, state_space.n))
        reward1 = np.zeros((state_space.n, human_action_space.n, robot_action_space.n, state_space.n))

        for h_st in range(human_state_space.n):
            for r_st in range(robot_state_space.n):
                st = state_idx(h_st, r_st)
                for h_a in range(human_action_space.n):
                    for r_a in range(robot_action_space.n):

                        for n_h_st in range(human_state_space.n):
                            for n_r_st in range(robot_state_space.n):
                                n_st = state_idx(n_h_st, n_r_st)

                                reward0[st, h_a, r_a, n_st] = (
                                    reward_0_h[h_st, h_a, n_h_st]
                                    + reward_0_r[r_st, r_a, n_r_st]
                                )
                                reward1[st, h_a, r_a, n_st] = (
                                    reward_1_h[h_st, h_a, n_h_st]
                                    + reward_1_r[r_st, r_a, n_r_st]
                                )

                                transition[st, h_a, r_a, n_st] = (
                                    human_transition[h_st, h_a, n_h_st]
                                    * robot_transition[r_st, r_a, n_r_st]
                                )


        rewards_dist = [(reward0, 0.5), (reward1, 0.5)]
        initial_state_dist = np.zeros(num_states)
        initial_state_dist[0] = 1.0

        horizon = 4
        discount = 0.9

        super().__init__(
            state_space=state_space,
            human_action_space=human_action_space,
            robot_action_space=robot_action_space,
            transition=transition,
            reward_distribution=rewards_dist,
            initial_state_distribution=initial_state_dist,
            horizon=horizon,
            discount=discount,
        )


class RedBlueAssistanceProblem(AssistanceProblem):
    def __init__(self, human_policy_fn=get_human_policy, use_belief_space=True):
        assistance_game = RedBlueAssistanceGame()

        if use_belief_space:
            observation_model_fn = BeliefObservationModel
        else:
            feature_extractor = lambda state : state % assistance_game.state_space.n
            setattr(feature_extractor, 'n', assistance_game.state_space.n)
            observation_model_fn = partial(DiscreteFeatureSenseObservationModel, feature_extractor=feature_extractor)

        reward_model_fn_builder = partial(discrete_reward_model_fn_builder, use_belief_space=use_belief_space)

        super().__init__(
            assistance_game=assistance_game,
            human_policy_fn=human_policy_fn,
            observation_model_fn=observation_model_fn,
            reward_model_fn_builder=reward_model_fn_builder,
        )
        self.ag_state_space_n = assistance_game.state_space.n

    def render(self, mode='human'):
        human_grid_pos = [(1, -1), (1, 0), (0, 0), (2, 0)]
        robot_grid_pos = [(1, 0), (0, 0), (2, 0)]

        if self.viewer is None:
            self.viewer = rendering.Viewer(500,600)
            self.viewer.set_bounds(-120, 120, -150, 120)

            grid_side = 30

            self.human_grid = rendering.Grid(start=(-110, 0), grid_side=grid_side, shape=(3, 1))
            self.human_grid_2 = rendering.Grid(start=(-80, -30), grid_side=grid_side, shape=(1, 1))
            self.viewer.add_geom(self.human_grid)
            self.viewer.add_geom(self.human_grid_2)

            self.robot_grid = rendering.Grid(start=(10, -15), grid_side=grid_side, shape=(3, 1))
            self.viewer.add_geom(self.robot_grid)


            human_red_ball_coords = self.human_grid.coords_from_pos(human_grid_pos[2])
            human_blue_ball_coords = self.human_grid.coords_from_pos(human_grid_pos[3])

            robot_red_ball_coords = self.robot_grid.coords_from_pos(robot_grid_pos[1])
            robot_blue_ball_coords = self.robot_grid.coords_from_pos(robot_grid_pos[2])

            red = (.8, .2, .2)
            blue = (.2, .2, .8)
            
            radius = grid_side * 1/3

            for coords, color in (
                (human_red_ball_coords, red),
                (human_blue_ball_coords, blue),
                (robot_red_ball_coords, red),
                (robot_blue_ball_coords, blue),
            ):
                ball = rendering.make_circle(radius)
                ball.set_color(*color)
                ball.add_attr(rendering.Transform(translation=coords))
                self.viewer.add_geom(ball)

            human_image = get_asset('images/girl1.png')
            human = rendering.Image(human_image, grid_side, grid_side)
            self.human_transform = rendering.Transform()
            human.add_attr(self.human_transform)
            self.viewer.add_geom(human)

            robot_image = get_asset('images/robot1.png')
            robot = rendering.Image(robot_image, grid_side, grid_side)
            self.robot_transform = rendering.Transform()
            robot.add_attr(self.robot_transform)
            self.viewer.add_geom(robot)

        ob = self.state % self.ag_state_space_n
        human_state = ob // 3
        robot_state = ob % 3

        human_coords = self.human_grid.coords_from_pos(human_grid_pos[human_state])
        robot_coords = self.robot_grid.coords_from_pos(robot_grid_pos[robot_state])

        self.human_transform.set_translation(*human_coords)
        self.robot_transform.set_translation(*robot_coords)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')


class WardrobeAssistanceGame(AssistanceGame):
    """

    This is the map of the assistance game:

    .......
    .     .
    .     .
    . TT  .
    . TT  .
    .H   R.
    .......

    H = Human
    R = Robot
    T = Wardrobe

    The human wants to move the wardrobe to one of the corners.
    To move the wardrobe, two agents need to push it at the same time
    in the same direction.

    Size of the map and of the table can be chosen.
    """

    State = namedtuple('State', ['human', 'robot', 'wardrobe'])

    def __init__(self, size=3, wardrobe_size=1):
        self.size = size
        self.wardrobe_size = wardrobe_size

        wardrobe_range = self.size - self.wardrobe_size + 1
        self.wardrobe_range = wardrobe_range

        target_and_reward =  {
            (0, 0) : {},
            (wardrobe_range - 1, 0) : {},
            (0, wardrobe_range - 1) : {},
            (wardrobe_range - 1, wardrobe_range - 1) : {},
        }
        self.targets = target_and_reward

        num_states = self.size ** 4 * self.wardrobe_range ** 2
        state_space = Discrete(num_states)

        num_actions = 4
        action_space = Discrete(num_actions)


        T = {}
        T_shape = (num_states, num_actions, num_actions, num_states)

        for hy in range(self.size):
            for hx in range(self.size):
                for ry in range(self.size):
                    for rx in range(self.size):
                        for ty in range(wardrobe_range):
                            for tx in range(wardrobe_range):
                                state = self.State(human=(hx, hy), robot=(rx, ry), wardrobe=(tx, ty))
                                idx = self.get_idx(state)
                                for ah in range(num_actions):
                                    for ar in range(num_actions):
                                        next_state = self.transition_fn(state, ah, ar)
                                        next_idx = self.get_idx(next_state)
                                        T[idx, ah, ar, next_idx] = 1.0
                                        if next_state.wardrobe in target_and_reward and state.wardrobe != next_state.wardrobe:
                                            target_and_reward[next_state.wardrobe][idx, ah, ar, next_idx] = 1.0

        transition = dict_to_sparse(T, T_shape)

        rewards_dist = [(dict_to_sparse(R, T_shape), 0.25) for R in target_and_reward.values()]

        initial_state = ((0, 0), (self.size - 1, 0), (1, 1))
        initial_state_dist = np.zeros(num_states)
        initial_state_dist[self.get_idx(initial_state)] = 1.0


        horizon = 3 * self.size
        discount = 0.7

        super().__init__(
            state_space=state_space,
            human_action_space=action_space,
            robot_action_space=action_space,
            transition=transition,
            reward_distribution=rewards_dist,
            initial_state_distribution=initial_state_dist,
            horizon=horizon,
            discount=discount,
        )


    def transition_fn(self, state, human_action, robot_action):
        next_human_pos = self.move(state.human, human_action)
        next_robot_pos = self.move(state.robot, robot_action)

        if (human_action == robot_action and
            self.wardrobe_can_move(state.wardrobe, human_action) and
            self.in_wardrobe(state, next_human_pos) and
            self.in_wardrobe(state, next_robot_pos)
        ):
            next_wardrobe_pos = self.move(state.wardrobe, human_action)
        else:
            next_wardrobe_pos = state.wardrobe
            next_human_pos = next_human_pos if not self.in_wardrobe(state, next_human_pos) else state.human
            next_robot_pos = next_robot_pos if not self.in_wardrobe(state, next_robot_pos) else state.robot

        return self.State(human=next_human_pos,
                          robot=next_robot_pos,
                          wardrobe=next_wardrobe_pos)


    def move(self, pos, act, size=None):
        if size is None:
            size = self.size

        x, y = pos
        dirs = [
            (1, 0),
            (0, 1),
            (-1, 0),
            (0, -1),
        ]
        dx, dy = dirs[act]

        new_x = np.clip(x + dx, 0, size - 1)
        new_y = np.clip(y + dy, 0, size - 1)

        return new_x, new_y

    def wardrobe_can_move(self, pos, act):
        return self.move(pos, act, size=self.wardrobe_range) != pos


    def in_wardrobe(self, state, pos):
        wardrobe_x, wardrobe_y = state.wardrobe
        x, y = pos
        return (0 <= x - wardrobe_x < self.wardrobe_size and
                0 <= y - wardrobe_y < self.wardrobe_size)

    def get_idx(self, state):
        (hx, hy), (rx, ry), (tx, ty) = state

        return (
            ty + self.wardrobe_range * (
            tx + self.wardrobe_range * (
            ry + self.size * (
            rx + self.size * (
            hy + self.size * (
            hx)))))
        )

    def get_state(self, idx):
        steps = [self.wardrobe_range, self.wardrobe_range, self.size, self.size, self.size, self.size]

        vals = []
        for step in steps:
            vals.append(idx % step)
            idx //= step

        hx, hy, rx, ry, tx, ty = reversed(vals)
        return self.State(human=(hx, hy), robot=(rx, ry), wardrobe=(tx, ty))


class WardrobeAssistanceProblem(AssistanceProblem):
    def __init__(self, human_policy_fn=get_human_policy, use_belief_space=True):
        self.assistance_game = WardrobeAssistanceGame()

        if use_belief_space:
            observation_model_fn = BeliefObservationModel
        else:
            feature_extractor = lambda state : state % self.assistance_game.state_space.n
            setattr(feature_extractor, 'n', self.assistance_game.state_space.n)
            observation_model_fn = partial(DiscreteFeatureSenseObservationModel, feature_extractor=feature_extractor)

        reward_model_fn_builder = partial(discrete_reward_model_fn_builder, use_belief_space=use_belief_space)

        super().__init__(
            assistance_game=self.assistance_game,
            human_policy_fn=human_policy_fn,
            observation_model_fn=observation_model_fn,
            reward_model_fn_builder=reward_model_fn_builder,
        )

    def render(self, mode='human'):
        size = self.assistance_game.size
        wardrobe_size = self.assistance_game.wardrobe_size

        if self.viewer is None:
            self.viewer = rendering.Viewer(500,600)
            self.viewer.set_bounds(-120, 120, -150, 120)

            self.grid = rendering.Grid(start=(-100, -100), end=(100, 100), shape=(size, size))
            self.viewer.add_geom(self.grid)

            human_image = get_asset('images/girl1.png')
            human = rendering.Image(human_image, self.grid.side, self.grid.side)
            self.human_transform = rendering.Transform()
            human.add_attr(self.human_transform)
            self.viewer.add_geom(human)

            robot_image = get_asset('images/robot1.png')
            robot = rendering.Image(robot_image, self.grid.side, self.grid.side)
            self.robot_transform = rendering.Transform()
            robot.add_attr(self.robot_transform)
            self.viewer.add_geom(robot)

            wardrobe_image = get_asset('images/wardrobe1.png')
            wardrobe = rendering.Image(wardrobe_image, wardrobe_size * self.grid.side, wardrobe_size * self.grid.side)
            self.wardrobe_transform = rendering.Transform()
            wardrobe.add_attr(self.wardrobe_transform)
            self.viewer.add_geom(wardrobe)

        nS0 = self.assistance_game.state_space.n
        idx = self.state % nS0
        rew_idx = self.state // nS0

        human_pos, robot_pos, wardrobe_top_left = self.assistance_game.get_state(idx)

        human_coords = self.grid.coords_from_pos(human_pos)
        robot_coords = self.grid.coords_from_pos(robot_pos)

        tl_x, tl_y = wardrobe_top_left
        wardrobe_bot_right = tl_x + wardrobe_size - 1, tl_y + wardrobe_size - 1

        tl_x, tl_y = self.grid.coords_from_pos(wardrobe_top_left)
        br_x, br_y = self.grid.coords_from_pos(wardrobe_bot_right)

        wardrobe_coords = ((tl_x + br_x) / 2, (tl_y + br_y) / 2)

        self.human_transform.set_translation(*human_coords)
        self.robot_transform.set_translation(*robot_coords)
        self.wardrobe_transform.set_translation(*wardrobe_coords)


        def add_bar(pos, ratio):
            x, y = self.grid.coords_from_pos(pos)
            xs, ys = self.grid.x_step, self.grid.y_step
            l, r = x - xs/2, x - xs/2 + 3
            b = y - ys/2
            t = b + ratio * ys
            bar = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            bar.set_color(0.7, 0.3, 0.3)
            self.viewer.add_onetime(bar)

        if hasattr(self.observation_model, 'belief'):
            reward_beliefs = self.observation_model.belief.reshape(-1, nS0).sum(axis=1)

            for pos, ratio in zip(self.assistance_game.targets, reward_beliefs):
                add_bar(pos, ratio)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')



class ChocolatePieGame(AssistanceGame):
    def __init__(self):
        human_state_space = Discrete(5)
        human_action_space = Discrete(3)

        robot_state_space = Discrete(4)
        robot_action_space = Discrete(2)

        nSh = human_state_space.n
        nAh = human_action_space.n
        nSr = robot_state_space.n
        nAr = robot_action_space.n

        nS = nSh * nSr
        state_space = Discrete(nS)

        T = np.zeros((nSh, nSr, nAh, nAr, nSh, nSr))

        R0 = np.zeros((nSh, nSr, nAh, nAr, nSh, nSr))
        R1 = np.zeros((nSh, nSr, nAh, nAr, nSh, nSr))

        for s_h, s_r, a_h, a_r in itertools.product(range(nSh), range(nSr), range(nAh), range(nAr)):
            n_s_h, n_s_r = self.transition_fn(s_h, s_r, a_h, a_r)
            T[s_h, s_r, a_h, a_r, n_s_h, n_s_r] = 1.0
            
        R0[0, :, 2, :, :, :] = -1.0
        R1[0, :, 2, :, :, :] = -1.0

        R0[:, 1, :, :, :, 2] = 10.0
        R0[:, 1, :, :, :, 3] = 1.0

        R1[:, 1, :, :, :, 2] = 1.0
        R1[:, 1, :, :, :, 3] = 10.0

        T = T.reshape(nS, nAh, nAr, nS)
        R0 = R0.reshape(nS, nAh, nAr, nS)
        R1 = R1.reshape(nS, nAh, nAr, nS)


        initial_state_dist = np.zeros(nS)
        initial_state_dist[0] = 1.0

        rewards_dist = [(R0, 0.5), (R1, 0.5)]

        horizon = 5
        discount = 1.0

        super().__init__(
            state_space=state_space,
            human_action_space=human_action_space,
            robot_action_space=robot_action_space,
            transition=T,
            reward_distribution=rewards_dist,
            initial_state_distribution=initial_state_dist,
            horizon=horizon,
            discount=discount,
        )

    @staticmethod
    def transition_fn(s_h, s_r, a_h, a_r):
        if s_h == 0:
            n_s_h = a_h + 1
        elif s_h == 3:
            n_s_h = 4
        else:
            n_s_h = s_h

        if s_r == 0 and n_s_h in (1, 2, 4):
            n_s_r = 1
        elif s_r == 1:
            n_s_r = 1 + a_r + 1
        else:
            n_s_r = s_r

        return n_s_h, n_s_r


def get_chocolate_pie_expert(assistance_game, reward):
    is_R0 = reward[1, 0, 0, 2] > 2.0

    P = 1/3 * np.ones(reward.shape[:2])
    if is_R0:
        P[0, 0] = 0.5
        P[0, 1] = 0.5
        P[0, 2] = 0.0
    else:
        P[0, 0] = 0.0
        P[0, 1] = 0.0
        P[0, 2] = 1.0
    
    return P


class ChocolateAssistanceProblem(AssistanceProblem):
    def __init__(self, human_policy_fn=get_chocolate_pie_expert, use_belief_space=True):
        assistance_game = ChocolatePieGame()

        if use_belief_space:
            observation_model_fn = BeliefObservationModel
        else:
            feature_extractor = lambda state : state % assistance_game.state_space.n
            setattr(feature_extractor, 'n', assistance_game.state_space.n)
            observation_model_fn = partial(DiscreteFeatureSenseObservationModel, feature_extractor=feature_extractor)

        reward_model_fn_builder = partial(discrete_reward_model_fn_builder, use_belief_space=use_belief_space)

        super().__init__(
            assistance_game=assistance_game,
            human_policy_fn=human_policy_fn,
            observation_model_fn=observation_model_fn,
            reward_model_fn_builder=reward_model_fn_builder,
        )
        self.ag_state_space_n = assistance_game.state_space.n


    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = rendering.Viewer(500,600)
            self.viewer.set_bounds(-120, 120, -150, 120)

            grid_side = 30

            self.grid = rendering.Grid(start=(-110, -110), grid_side=grid_side, shape=(7, 6))
            self.grid.set_color(0.85, 0.85, 0.85)
            self.viewer.add_geom(self.grid)

            human_image = get_asset('images/girl1.png')
            human = rendering.Image(human_image, grid_side, grid_side)
            self.human_transform = rendering.Transform()
            human.add_attr(self.human_transform)
            self.viewer.add_geom(human)

            robot_image = get_asset('images/robot1.png')
            robot = rendering.Image(robot_image, grid_side, grid_side)
            self.robot_transform = rendering.Transform()
            robot.add_attr(self.robot_transform)
            self.viewer.add_geom(robot)

            robot_coords = self.grid.coords_from_pos((4, 3))
            self.robot_transform.set_translation(*robot_coords)


            make_rect = lambda x, y, w, h : rendering.make_polygon([(x,y),(x+w,y),(x+w,y+h),(x,y+h)])
            gs = grid_side
            make_grid_rect = lambda i, j, di, dj : make_rect(-110 + i*gs, -110 + j*gs, di*gs, dj*gs)

            # Top counter  
            counters = [
                make_grid_rect(0, 0, 1, 6),
                make_grid_rect(3, 0, 1, 6),
                make_grid_rect(6, 0, 1, 6),
                make_grid_rect(0, 0, 7, 1),
                make_grid_rect(0, 5, 7, 1),
            ]
            for counter in counters:
                r = 0.8
                off = 0.05
                g = r - off
                b = r - 2 * off
                counter.set_color(r, g, b)
                self.viewer.add_geom(counter)

            flour_image = get_asset('images/flour3.png')
            scale = 0.6
            flour = rendering.Image(flour_image, scale * grid_side, scale * grid_side)
            flour_transform = rendering.Transform()
            flour.add_attr(flour_transform)
            self.viewer.add_geom(flour)

            flour_coords = self.grid.coords_from_pos((0, 3))
            flour_transform.set_translation(*flour_coords)


            apple_image = get_asset('images/apple3.png')
            scale = 0.6
            apple = rendering.Image(apple_image, scale * grid_side, scale * grid_side)
            apple.set_color(0.5, 0.7, 0.0)
            apple_transform = rendering.Transform()
            apple.add_attr(apple_transform)
            self.viewer.add_geom(apple)

            apple_coords = self.grid.coords_from_pos((2, 5))
            apple_transform.set_translation(*apple_coords)


            chocolate_image = get_asset('images/chocolate2.png')
            scale = 0.7
            chocolate = rendering.Image(chocolate_image, scale * grid_side, scale * grid_side)
            chocolate_transform = rendering.Transform()
            chocolate.add_attr(chocolate_transform)
            self.viewer.add_geom(chocolate)

            chocolate_coords = self.grid.coords_from_pos((2, 0))
            chocolate_transform.set_translation(*chocolate_coords)


            applepie_image = get_asset('images/apple-pie2.png')
            scale = 0.7
            applepie = rendering.Image(applepie_image, scale * grid_side, scale * grid_side)
            applepie_transform = rendering.Transform()
            applepie.add_attr(applepie_transform)

            applepie_coords = self.grid.coords_from_pos((3, 3))
            applepie_transform.set_translation(*applepie_coords)


            chocpie_image = get_asset('images/chocolate-pie2.png')
            scale = 0.7
            chocpie = rendering.Image(chocpie_image, scale * grid_side, scale * grid_side)
            chocpie_transform = rendering.Transform()
            chocpie.add_attr(chocpie_transform)

            chocpie_coords = self.grid.coords_from_pos((3, 3))
            chocpie_transform.set_translation(*chocpie_coords)

            self.applepie = applepie
            self.chocpie = chocpie




            hl = 15
            header_x = -15 + hl
            header_y = -110 + 6 * grid_side + hl

            scale = 0.4
            flour2 = rendering.Image(flour_image, scale * grid_side, scale * grid_side)
            flour2_transform = rendering.Transform()
            flour2.add_attr(flour2_transform)
            self.viewer.add_geom(flour2)

            flour2_transform.set_translation(header_x, header_y)


            plus_image = get_asset('images/plus1.png')

            scale = 0.2
            plus1 = rendering.Image(plus_image, scale * grid_side, scale * grid_side)
            plus1_transform = rendering.Transform()
            plus1.add_attr(plus1_transform)
            self.viewer.add_geom(plus1)

            plus1_transform.set_translation(header_x + 1*hl, header_y)

            scale = 0.2
            plus2 = rendering.Image(plus_image, scale * grid_side, scale * grid_side)
            plus2_transform = rendering.Transform()
            plus2.add_attr(plus2_transform)
            self.viewer.add_geom(plus2)

            plus2_transform.set_translation(header_x + 3*hl, header_y)

            scale = 0.2
            plus2 = rendering.Image(plus_image, scale * grid_side, scale * grid_side)
            plus2_transform = rendering.Transform()
            plus2.add_attr(plus2_transform)
            self.viewer.add_geom(plus2)

            plus2_transform.set_translation(header_x + 1*hl, header_y + 1.2*hl)



            equal_image = get_asset('images/equal1.png')

            scale = 0.15
            equal1 = rendering.Image(equal_image, scale * grid_side, scale * grid_side)
            equal1_transform = rendering.Transform()
            equal1.add_attr(equal1_transform)
            self.viewer.add_geom(equal1)

            equal1_transform.set_translation(header_x + 5*hl, header_y)

            scale = 0.15
            equal2 = rendering.Image(equal_image, scale * grid_side, scale * grid_side)
            equal2_transform = rendering.Transform()
            equal2.add_attr(equal2_transform)
            self.viewer.add_geom(equal2)

            equal2_transform.set_translation(header_x + 3*hl, header_y + 1.2*hl)



            scale = 0.4
            apple2 = rendering.Image(apple_image, scale * grid_side, scale * grid_side)
            apple2.set_color(0.5, 0.7, 0.0)
            apple2_transform = rendering.Transform()
            apple2.add_attr(apple2_transform)
            self.viewer.add_geom(apple2)

            apple2_transform.set_translation(header_x + 2*hl, header_y)


            scale = 0.4
            chocolate2 = rendering.Image(chocolate_image, scale * grid_side, scale * grid_side)
            chocolate2_transform = rendering.Transform()
            chocolate2.add_attr(chocolate2_transform)
            self.viewer.add_geom(chocolate2)

            chocolate2_transform.set_translation(header_x + 4*hl, header_y)



            scale = 0.4
            flour3 = rendering.Image(flour_image, scale * grid_side, scale * grid_side)
            flour2_transform = rendering.Transform()
            flour3.add_attr(flour2_transform)
            self.viewer.add_geom(flour3)

            flour2_transform.set_translation(header_x, header_y + 1.2*hl)


            scale = 0.4
            apple3 = rendering.Image(apple_image, scale * grid_side, scale * grid_side)
            apple3.set_color(0.5, 0.7, 0.0)
            apple3_transform = rendering.Transform()
            apple3.add_attr(apple3_transform)
            self.viewer.add_geom(apple3)

            apple3_transform.set_translation(header_x + 2*hl, header_y + 1.2*hl)


            scale = 0.4
            applepie2 = rendering.Image(applepie_image, scale * grid_side, scale * grid_side)
            applepie2_transform = rendering.Transform()
            applepie2.add_attr(applepie2_transform)
            self.viewer.add_geom(applepie2)

            applepie2_transform.set_translation(header_x + 4*hl, header_y + 1.2*hl)


            scale = 0.4
            chocpie2 = rendering.Image(chocpie_image, scale * grid_side, scale * grid_side)
            chocpie2_transform = rendering.Transform()
            chocpie2.add_attr(chocpie2_transform)
            self.viewer.add_geom(chocpie2)

            chocpie2_transform.set_translation(header_x + 6*hl, header_y)


            scale = 1.0
            thought = rendering.make_ellipse(scale * grid_side/2, scale * 0.7*grid_side/2)
            thought.set_color(0.9, 0.9, 0.9)
            self.thought_transform = rendering.Transform()
            thought.add_attr(self.thought_transform)

            self.viewer.add_geom(thought)

            scale = 0.17
            thought2 = rendering.make_ellipse(scale * grid_side/2, scale * grid_side/2)
            thought2.set_color(0.9, 0.9, 0.9)
            self.thought_transform2 = rendering.Transform()
            thought2.add_attr(self.thought_transform2)
            self.viewer.add_geom(thought2)

            scale = 0.1
            thought3 = rendering.make_ellipse(scale * grid_side/2, scale * grid_side/2)
            thought3.set_color(0.9, 0.9, 0.9)
            self.thought_transform3 = rendering.Transform()
            thought3.add_attr(self.thought_transform3)
            self.viewer.add_geom(thought3)



            hl2 = 20
            header2_x = -110 + hl2
            header2_y = -110 + 6 * grid_side + hl2

            scale = 0.3
            applepie3 = rendering.Image(applepie_image, scale * grid_side, scale * grid_side)
            applepie3_transform = rendering.Transform()
            applepie3.add_attr(applepie3_transform)
            self.viewer.add_geom(applepie3)

            applepie3_transform.set_translation(header2_x, header2_y)
            self.tgt_apple_transform = applepie3_transform

            scale = 0.3
            chocpie3 = rendering.Image(chocpie_image, scale * grid_side, scale * grid_side)
            chocpie3_transform = rendering.Transform()
            chocpie3.add_attr(chocpie3_transform)
            self.viewer.add_geom(chocpie3)

            chocpie3_transform.set_translation(header2_x + 2*hl2, header2_y)
            self.tgt_choc_transform = chocpie3_transform


            comp_transform = rendering.Transform()
            self.comp_transform = comp_transform

            greater_image = get_asset('images/greater1.png')

            scale = 0.15
            greater1 = rendering.Image(greater_image, scale * grid_side, scale * grid_side)
            greater1.add_attr(comp_transform)

            less_image = get_asset('images/less1.png')

            scale = 0.15
            less1 = rendering.Image(less_image, scale * grid_side, scale * grid_side)
            less1.add_attr(comp_transform)


            self.greater = greater1
            self.less = less1




        human_grid_pos = [
            (2, 3),
            (1, 3),
            (2, 4),
            (2, 2),
            (2, 1),
        ]

        reward_idx = self.state // self.ag_state_space_n
        state = self.state % self.ag_state_space_n

        human_state = state // 4
        robot_state = state % 4


        human_pos = human_grid_pos[human_state]

        human_coords = self.grid.coords_from_pos(human_pos)
        self.human_transform.set_translation(*human_coords)

        thought_pos = (lambda x, y : (x-1,y+1))(*human_pos)
        thought_coords = self.grid.coords_from_pos(thought_pos)
        self.thought_transform.set_translation(*thought_coords)
        thought_coords2 = (lambda x, y : (x+12,y-12))(*thought_coords)
        self.thought_transform2.set_translation(*thought_coords2)
        thought_coords3 = (lambda x, y : (x+17,y-17))(*thought_coords)
        self.thought_transform3.set_translation(*thought_coords3)

        tgt_apple_coords = (lambda x, y : (x-8,y))(*thought_coords)
        tgt_comp_coords = thought_coords
        tgt_choc_coords = (lambda x, y : (x+8,y))(*thought_coords)

        self.tgt_apple_transform.set_translation(*tgt_apple_coords)
        self.comp_transform.set_translation(*tgt_comp_coords)
        self.tgt_choc_transform.set_translation(*tgt_choc_coords)

        if robot_state == 2:
            self.viewer.add_onetime(self.applepie)
        elif robot_state == 3:
            self.viewer.add_onetime(self.chocpie)

        if reward_idx == 0:
            self.viewer.add_onetime(self.greater)
        else:
            self.viewer.add_onetime(self.less)


        print(reward_idx, human_state, robot_state)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')


class PlateAssistanceGame(AssistanceGame):
    def __init__(self):
        self.width = 5
        self.height = 6

        self.counter_items = {
            (0, 2) : 'A',
            (4, 5) : 'B',
            (0, 4) : 'C',
            (0, 5) : 'D',
            (4, 1) : 'E',
            (4, 2) : 'F',
            (2, 5) : 'P',
        }


        self.recipes = [
            ('A', 'A', 'B', 'B'),
            ('A', 'B', 'C', 'E'),
            ('A', 'B', 'D', 'F'),
        ]

        human_action_space = Discrete(6)
        robot_action_space = Discrete(6)

        self.INTERACT = 5

        state_space = None
        self.initial_state = {
            'human_pos' : (0, 2),
            'human_hand' : '',
            'robot_pos' : (4, 2),
            'robot_hand' : '',
            'plate' : (),
        }

        horizon = 20
        discount = 0.9

        rewards_dist = []
        num_recipes = len(self.recipes)
        for idx in range(num_recipes):
            reward_fn = partial(self.reward_fn, reward_idx=idx)
            rewards_dist.append((reward_fn, 1/num_recipes))
        

        super().__init__(
            state_space=state_space,
            human_action_space=human_action_space,
            robot_action_space=robot_action_space,
            transition=self.transition_fn,
            reward_distribution=rewards_dist,
            horizon=horizon,
            discount=discount,
        )

    def reward_fn(self, state, next_state=None, human_action=0, robot_action=0, reward_idx=0):
        recipe_reward = int(state['plate'] == self.recipes[reward_idx])
        return recipe_reward

    def transition_fn(self, state, human_action=0, robot_action=0):
        s = state.copy()

        s['human_pos'] = self.update_pos(state['human_pos'], human_action)
        s['robot_pos'] = self.update_pos(state['robot_pos'], robot_action)

        s['human_hand'] = self.update_hand(state['human_pos'], state['human_hand'], human_action)
        s['robot_hand'] = self.update_hand(state['robot_pos'], state['robot_hand'], robot_action)

        s['plate'] = self.update_plate(state['plate'], state['human_pos'], state['human_hand'], human_action)
        s['plate'] = self.update_plate(s['plate'], state['robot_pos'], state['robot_hand'], robot_action)

        return s


    def update_pos(self, pos, act):
        x, y = pos
        dirs = [
            (0, 0),
            (1, 0),
            (0, 1),
            (-1, 0),
            (0, -1),
            (0, 0),
        ]
        dx, dy = dirs[act]

        new_x = np.clip(x + dx, 0, self.width - 1)
        new_y = np.clip(y + dy, 0, self.height - 1)

        return new_x, new_y


    def update_hand(self, pos, hand, action):
        if action == self.INTERACT and pos in self.counter_items:
            if hand == '' and self.counter_items[pos] != 'P':
                return self.counter_items[pos]
            if self.counter_items[pos] in (hand, 'P'):
                return ''
        return hand


    def update_plate(self, plate, pos, hand, action):
        if hand != '' and action == self.INTERACT and self.counter_items.get(pos, '') == 'P' and len(plate) < 4:
            new_plate = list(plate)
            new_plate.append(hand)
            new_plate = tuple(sorted(new_plate))
            return new_plate
        return plate


class PlateAssistanceProblem(AssistanceProblem):
    def __init__(self, human_policy_fn=functional_random_policy_fn, **kwargs):
        assistance_game = PlateAssistanceGame()

        human_policy_fn = plate_human_policy_fn

        self.ag = assistance_game

        super().__init__(
            assistance_game=assistance_game,
            human_policy_fn=human_policy_fn,

            state_space_builder=plate_state_space_builder,
            transition_model_fn_builder=plate_transition_model_fn_builder,
            reward_model_fn_builder=plate_reward_model_fn_builder,
            sensor_model_fn_builder=plate_sensor_model_fn_builder,
            observation_model_fn=plate_observation_model_fn_builder(assistance_game),
        )

    def render(self, mode='human'):
        print(self.state)

        width = 5
        height = 6


        grid_side = 30
        gs = grid_side


        def make_image_transform(filename, w=1.0, h=None, s=0.6, c=None):
            if h is None: h = w
            fullname = get_asset(f'images/{filename}')
            obj = rendering.Image(fullname, s * w * gs, s * h * gs)
            transform = rendering.Transform()
            obj.add_attr(transform)

            if c is not None:
                obj.set_color(*c)

            return obj, transform

        make_item_image = {
            'A' : partial(make_image_transform, 'flour3.png'),
            'B' : partial(make_image_transform, 'flour3.png', c=(0.1, 0.1, 0.1)),
            'C' : partial(make_image_transform, 'apple3.png', c=(0.5, 0.7, 0.0)),
            'D' : partial(make_image_transform, 'apple3.png', c=(0.7, 0.3, 0.2)),
            'E' : partial(make_image_transform, 'chocolate2.png'),
            'F' : partial(make_image_transform, 'chocolate2.png', c=(0.1, 0.1, 0.1)),
            'P' : partial(make_image_transform, 'plate1.png', w=1.3, h=1.3),
            '+' : partial(make_image_transform, 'plus1.png', w=0.5, h=0.5),
            '=' : partial(make_image_transform, 'equal1.png', w=0.5, h=0.2),
            '0' : partial(make_image_transform, 'apple-pie1.png'),
            '1' : partial(make_image_transform, 'apple-pie1.png', c=(0.5, 0.7, 0.0)),
            '2' : partial(make_image_transform, 'apple-pie1.png', c=(0.7, 0.3, 0.2)),
        }

        def move_to_counter(pos):
            x, y = pos
            if x == 0:
                return (-1, y)
            elif y == 0:
                return (x, -1)
            elif x == width - 1:
                return (width, y)
            elif y == height - 1:
                return (x, height)
            else:
                return (x, y)


        if self.viewer is None:
            self.viewer = rendering.Viewer(500,800)
            self.viewer.set_bounds(-130, 120, -150, 250)

            g_x0 = -110 + grid_side
            g_y0 = -110 + grid_side

            self.grid = rendering.Grid(start=(g_x0, g_y0), grid_side=grid_side, shape=(width, height))
            self.grid.set_color(0.85, 0.85, 0.85)
            self.viewer.add_geom(self.grid)

            human_image = get_asset('images/girl1.png')
            human = rendering.Image(human_image, grid_side, grid_side)
            self.human_transform = rendering.Transform()
            human.add_attr(self.human_transform)
            self.viewer.add_geom(human)

            robot_image = get_asset('images/robot1.png')
            robot = rendering.Image(robot_image, grid_side, grid_side)
            self.robot_transform = rendering.Transform()
            robot.add_attr(self.robot_transform)
            self.viewer.add_geom(robot)


            make_rect = lambda x, y, w, h : rendering.make_polygon([(x,y),(x+w,y),(x+w,y+h),(x,y+h)])

            make_grid_rect = lambda i, j, di, dj : make_rect(g_x0 + i*gs, g_y0 + j*gs, di*gs, dj*gs)

            counters = [
                make_grid_rect(-1, -1, width+2, 1),
                make_grid_rect(-1, -1, 1, height+2),

                make_grid_rect(-1, height, width+2, 1),
                make_grid_rect(width, -1, 1, height+2),
            ]

            for counter in counters:
                r = 0.8
                off = 0.05
                g = r - off
                b = r - 2 * off
                counter.set_color(r, g, b)
                self.viewer.add_geom(counter)

            for pos, itemname in self.ag.counter_items.items():
                counter_pos = move_to_counter(pos)
                coords = self.grid.coords_from_pos(counter_pos)

                item, transform = make_item_image[itemname]()
                transform.set_translation(*coords)
                print(item, transform, coords)
                self.viewer.add_geom(item)


            # Render formulae

            g_x0 = -110 + grid_side
            g_y0 = -110 + grid_side
            header_x = g_x0 + 0 * grid_side
            header_y = g_y0 + (height + 2) * grid_side
            hl = 15

            header_transform = rendering.Transform()
            header_transform.set_translation(header_x, header_y)

            for i, recipe in enumerate(self.ag.recipes):
                formula = '+'.join(recipe) + f'={i}'
                for j, c in enumerate(formula):
                    img, tr = make_item_image[c](s=0.5)
                    img.add_attr(header_transform)
                    tr.set_translation(hl*j, 1.2*hl*i)
                    self.viewer.add_geom(img)


            # Add thought balloon
            scale = 0.8
            thought = rendering.make_ellipse(scale * grid_side/2, scale * 0.7*grid_side/2)
            thought.set_color(0.9, 0.9, 0.9)
            thought_transform = rendering.Transform()
            thought_transform.set_translation(-gs, gs)
            thought.add_attr(thought_transform)
            thought.add_attr(self.human_transform)

            self.viewer.add_geom(thought)

            scale = 0.17
            thought2 = rendering.make_ellipse(scale * grid_side/2, scale * grid_side/2)
            thought2.set_color(0.9, 0.9, 0.9)
            thought_transform2 = rendering.Transform()
            thought_transform2.set_translation(-0.6 * gs, 0.6 * gs)
            thought2.add_attr(thought_transform2)
            thought2.add_attr(self.human_transform)

            self.viewer.add_geom(thought2)

            scale = 0.1
            thought3 = rendering.make_ellipse(scale * grid_side/2, scale * grid_side/2)
            thought3.set_color(0.9, 0.9, 0.9)
            thought_transform3 = rendering.Transform()
            thought_transform3.set_translation(-0.4 * gs, 0.4 * gs)
            thought3.add_attr(thought_transform3)
            thought3.add_attr(self.human_transform)

            self.viewer.add_geom(thought3)

            self.pies = []
            for pie_idx in ('012'):
                pie, _ = make_item_image[pie_idx](s=0.4)
                pie.add_attr(self.human_transform)
                pie.add_attr(thought_transform)
                self.pies.append(pie)


        human_pos = self.state['human_pos']
        robot_pos = self.state['robot_pos']
        human_hand = self.state['human_hand']
        robot_hand = self.state['robot_hand']
        plate = self.state['plate']
        reward_idx = self.state['reward_idx']

        human_coords = self.grid.coords_from_pos(human_pos)
        self.human_transform.set_translation(*human_coords)

        robot_coords = self.grid.coords_from_pos(robot_pos)
        self.robot_transform.set_translation(*robot_coords)

        for hand, hand_transform in (
            (human_hand, self.human_transform),
            (robot_hand, self.robot_transform),
        ):
            if hand != '':
                item, transform = make_item_image[hand](s=0.4)
                transform.set_translation(0, -5)
                item.add_attr(hand_transform)
                self.viewer.add_onetime(item)

        items_to_pos = {item:pos for pos, item in self.ag.counter_items.items()}
        plate_pos = move_to_counter(items_to_pos['P'])
        plate_coords = self.grid.coords_from_pos(plate_pos)

        recipe_made = False
        for idx, recipe in enumerate(self.ag.recipes):
            if plate == recipe:
                pie, transform = make_item_image[str(idx)](s=0.65)
                transform.set_translation(*plate_coords)
                self.viewer.add_onetime(pie)
                recipe_made = True
                break


        if not recipe_made:
            for j, item_name in enumerate(plate):
                item, transform = make_item_image[item_name](s=0.4)

                d = 7
                dx = (-1) ** (j+1) * d
                dy = (-1) ** (j // 2) * d
                item_coords = (lambda x, y : (x+dx, y+dy))(*plate_coords)

                transform.set_translation(*item_coords)
                self.viewer.add_onetime(item)


        self.viewer.add_onetime(self.pies[reward_idx])


        return self.viewer.render(return_rgb_array = mode=='rgb_array')


class PlateProblemStateSpace:
    def __init__(self, ag):
        self.initial_state = ag.initial_state
        self.num_rewards = len(ag.reward_distribution)

    def sample_initial_state(self):
        state = self.initial_state.copy()
        state['reward_idx'] = np.random.randint(self.num_rewards)
        return state

def plate_state_space_builder(ag):
    return PlateProblemStateSpace(ag)

def plate_transition_model_fn_builder(ag, human_policy_fn):
    def transition_fn(state, action):
        human_policy = human_policy_fn(ag, state['reward_idx'])
        return ag.transition(state, human_policy(state), action)

    transition_model_fn = partial(FunctionalTransitionModel, fn=transition_fn)
    return transition_model_fn

def plate_reward_model_fn_builder(ag, human_policy_fn):
    def reward_fn(state, action=None, next_state=None):
        reward = ag.reward_distribution[state['reward_idx']][0]
        return reward(state=state, next_state=next_state)

    reward_model_fn = partial(FunctionalRewardModel, fn=reward_fn)
    return reward_model_fn

def plate_sensor_model_fn_builder(ag, human_policy_fn):
    return SensorModel

def plate_observation_model_fn_builder(ag):
    num_ingredients = len(ag.counter_items) - 1

    def observation_fn(state, action=None, sense=None):
        def one_hot(i, n):
            return np.eye(n)[i]

        def position_ob(pos):
            x, y = pos
            return np.concatenate([
                one_hot(x, ag.width),
                one_hot(y, ag.height),
            ])

        def item_idx(item):
            return ord(item) - ord('A')

        def hand_ob(hand):
            if hand == '':
                return np.zeros(num_ingredients)
            else:
                return one_hot(item_idx(hand), num_ingredients)

        def plate_ob(plate):
            ob = np.zeros(num_ingredients)
            for item in plate:
                ob[item_idx(item)] = 1.0
            return ob

        return np.concatenate([
            position_ob(state['human_pos']),
            position_ob(state['robot_pos']),
            hand_ob(state['human_hand']),
            hand_ob(state['robot_hand']),
            plate_ob(state['plate']),
        ])

    num_dims = 4 + 3 * num_ingredients
    low = np.zeros(num_dims)
    high = np.ones(num_dims)
    high[:4] = [ag.width, ag.height, ag.width, ag.height]

    ob_space = Box(low=low, high=high)

    observation_model_fn = partial(FunctionalObservationModel, fn=observation_fn, space=ob_space)
    return observation_model_fn


def get_plate_hardcoded_robot_policy(*args, **kwargs):
    class Policy:
        def predict(self, ob, state=None):
            t, r_idx = state if state is not None else (0, None)

            width = 5
            height = 6

            onehot_x = ob[:width]
            onehot_y = ob[width:width+height]

            x = np.argmax(onehot_x)
            y = np.argmax(onehot_y)
            human_pos = (x, y)


            if t == 1:
                if human_pos == (0, 2):
                    r_idx = 0
                else:
                    r_idx = 3

            if r_idx == 3:
                if t == 3 and human_pos == (0, 4):
                    r_idx = 1

                if t == 3 and human_pos == (0, 5):
                    r_idx = 2



            S, R, U, L, D, A = range(6)


            robot_policies = [
                [
                    S,          # Wait 1 step
                    U, U, U, A, # Get dark flour
                    L, L, A,    # Take to plate
                    R, R, A,    # Get dark flour
                    L, L, A,    # Take to plate
                ],

                [
                    S, S, S,             # Wait 1 step
                    D, A,                # get milk chocolate
                    U, U, U, U, L, L, A, # drop in plate
                    R, R, A,             # get dark flour
                    L, L, A,             # Take to plate
                ],

                [
                    S, S, S,                # Wait 1 step
                    A,                      # get milk chocolate
                    U, U, U, L, L, A, # drop in plate
                    R, R, A,                # get dark flour
                    L, L, A,                # Take to plate
                ],
            ]

            if r_idx is None:
                robot_policy = robot_policies[0]
            elif r_idx == 3:
                robot_policy = robot_policies[1]
            else:
                robot_policy = robot_policies[r_idx]


            act = robot_policy[t] if t < len(robot_policy) else S

            return act, (t+1, r_idx)

    return Policy()


def plate_human_policy_fn(ag, reward):
    def human_policy(state):
        S, R, U, L, D, A = range(6)

        # policy 0

        policy0 = [
            A,                # get white flour
            U, U, U, R, R, A, # take to plate
            L, L, D, D, D, A, # get white flour
            U, U, U, R, R, A, # take to plate
            S,
        ]

        trail0 = [
            ((0, 2), '', {'A' : 0}),

            ((0, 2), 'A', {'A' : 0}),
            ((0, 3), 'A', {'A' : 0}),
            ((0, 4), 'A', {'A' : 0}),
            ((0, 5), 'A', {'A' : 0}),
            ((1, 5), 'A', {'A' : 0}),
            ((2, 5), 'A', {'A' : 0}),

            ((2, 5), '', {'A' : 1}),
            ((1, 5), '', {'A' : 1}),
            ((0, 5), '', {'A' : 1}),
            ((0, 4), '', {'A' : 1}),
            ((0, 3), '', {'A' : 1}),
            ((0, 2), '', {'A' : 1}),

            ((0, 2), 'A', {'A' : 1}),
            ((0, 3), 'A', {'A' : 1}),
            ((0, 4), 'A', {'A' : 1}),
            ((0, 5), 'A', {'A' : 1}),
            ((1, 5), 'A', {'A' : 1}),
            ((2, 5), 'A', {'A' : 1}),

            ((2, 5), '', {'A' : 2}),
        ]


        policy1 = [
            U, U, A, # get green apple
            U, R, R, A, # drop in plate
            L, L, D, D, D, A, # get white flour
            U, U, U, R, R, A, # take to plate
            S,
        ]

        trail1 = [
            ((0, 2), '',  {'C' : 0}),
            ((0, 3), '',  {'C' : 0}),
            ((0, 4), '',  {'C' : 0}),
            ((0, 4), 'C', {'C' : 0}),
            ((0, 5), 'C', {'C' : 0}),
            ((1, 5), 'C', {'C' : 0}),
            ((2, 5), 'C', {'C' : 0}),

            ((2, 5), '', {'A' : 0, 'C' : 1}),
            ((1, 5), '', {'C' : 1}),
            ((0, 5), '', {'C' : 1}),
            ((0, 4), '', {'C' : 1}),
            ((0, 3), '', {'C' : 1}),
            ((0, 2), '', {'C' : 1}),

            ((0, 2), 'A', {'C' : 1}),
            ((0, 3), 'A', {'C' : 1}),
            ((0, 4), 'A', {'C' : 1}),
            ((0, 5), 'A', {'C' : 1}),
            ((1, 5), 'A', {'C' : 1}),
            ((2, 5), 'A', {'C' : 1}),

            ((2, 5), '', {'A' : 1, 'C' : 1}),
        ]



        policy2 = [
            U, U, U, A, # get red apple
            R, R, A, # drop in plate
            L, L, D, D, D, A, # get white flour
            U, U, U, R, R, A, # take to plate
            S,
        ]

        trail2 = [
            ((0, 2), '',  {'A' : 0, 'D' : 0}),
            ((0, 3), '',  {'A' : 0, 'D' : 0}),
            ((0, 4), '',  {'A' : 0, 'D' : 0}),
            ((0, 5), '',  {'A' : 0, 'D' : 0}),
            ((0, 5), 'D', {'A' : 0, 'D' : 0}),
            ((1, 5), 'D', {'A' : 0, 'D' : 0}),
            ((2, 5), 'D', {'A' : 0, 'D' : 0}),

            ((2, 5), '', {'A' : 0, 'D' : 1}),
            ((1, 5), '', {'D' : 1}),
            ((0, 5), '', {'D' : 1}),
            ((0, 4), '', {'D' : 1}),
            ((0, 3), '', {'D' : 1}),
            ((0, 2), '', {'D' : 1}),

            ((0, 2), 'A', {'D' : 1}),
            ((0, 3), 'A', {'D' : 1}),
            ((0, 4), 'A', {'D' : 1}),
            ((0, 5), 'A', {'D' : 1}),
            ((1, 5), 'A', {'D' : 1}),
            ((2, 5), 'A', {'D' : 1}),

            ((2, 5), '', {'A' : 1, 'D' : 1}),
        ]


        reward_idx = state['reward_idx']

        policy = (policy0, policy1, policy2)[reward_idx]
        trail = (trail0, trail1, trail2)[reward_idx]


        def has_plate_condition(plate, cond):
            c = Counter(plate)
            return all(c[k] == v for k, v in cond.items())


        def get_time(state):
            for t, (pos, hand, plate_cond) in enumerate(trail):
                if (
                    pos == state['human_pos'] and
                    hand == state['human_hand'] and
                    has_plate_condition(state['plate'], plate_cond)
                ):
                    return t

        t = get_time(state)
        action = policy[t] if t is not None else A

        return action

    return human_policy
