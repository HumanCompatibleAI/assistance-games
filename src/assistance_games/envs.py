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
            observation_model_fn = partial(FeatureSenseObservationModel, feature_extractor=feature_extractor)

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
            observation_model_fn = partial(FeatureSenseObservationModel, feature_extractor=feature_extractor)

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


def query_response_cake_pizza(assistance_game, reward, **kwargs):
    ag = assistance_game
    num_states = ag.state_space.n
    n_world_states = ag.n_world_states
    num_actions = ag.human_action_space.n
    policy = np.zeros((num_states, num_actions))

    # Hardcoded query response for the simple CakePizza AG.
    # There are two actions available to the human; if r_pizza > r_cake
    # the human performs action 0, and otherwise performs action 1.
    policy[0:n_world_states, 0] = 1
    print(reward[2, 0, 0, 0], reward[3, 0, 0, 0])
    if reward[2, 0, 0, 0]>reward[3, 0, 0, 0]:
        policy[n_world_states:, 0] = 1
    else:
        policy[n_world_states:, 1] = 1
    return policy


def s_reward_to_saas_reward(state_reward, num_human_actions, num_robot_actions):
    "state_reward is a 1d vector of length equal to the number of states"
    num_states = len(state_reward)
    reward = np.zeros((num_states, num_human_actions, num_robot_actions, num_states))
    for s in range(num_states):
        reward[s, :, :, :] = state_reward[s]
    return reward


class CakePizzaGraphGame(AssistanceGame):
    def __init__(self):
        n_world_states = 4
        self.n_world_states = n_world_states
        n_world_actions = 2
        n_queries = 1
        self.n_queries = n_queries

        state_space = Discrete(n_world_states + n_world_states * n_queries)

        human_action_space = Discrete(n_queries * 2)
        robot_action_space = Discrete(n_world_actions + n_queries)

        # mdp transitions for world states: only robot can act, human's action doesn't matter
        transition = np.zeros((state_space.n, human_action_space.n, robot_action_space.n, state_space.n))
        transition[0, :, 0, 1] = 1
        transition[0, :, 1, 1] = 1
        transition[1, :, 0, 2] = 1
        transition[1, :, 1, 3] = 1
        transition[2, :, 0, 2] = 1
        transition[2, :, 1, 2] = 1
        transition[3, :, 0, 3] = 1
        transition[3, :, 1, 3] = 1

        # add transitions for the query actions of the robot, and the human response actions in the
        # n_states*n_queries information states
        for s in range(n_world_states):
            for q in range(n_queries):
                state_idx = n_world_states + s * (q + 1)
                robot_action_idx = n_world_actions + q

                # transitions to the query states
                transition[s, :, robot_action_idx, state_idx] = 1
                # transitions back to the world states
                transition[state_idx, :, :, s] = 1

        reward0 = np.zeros(state_space.n)
        reward0[0], reward0[1] = -0.2, -0.1
        reward0[2], reward0[3] = 2, 1
        reward0 = s_reward_to_saas_reward(reward0, human_action_space.n, robot_action_space.n)

        reward1 = np.zeros(state_space.n)
        reward1[0], reward1[1] = -0.2, -0.1
        reward1[2], reward1[3] = 1, 2
        reward1 = s_reward_to_saas_reward(reward1, human_action_space.n, robot_action_space.n)

        rewards_dist = [(reward0, 0.5), (reward1, 0.5)]
        initial_state_dist = np.zeros(state_space.n)
        initial_state_dist[0] = 1.0

        horizon = 10
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


class CakePizzaGraphProblem(AssistanceProblem):
    def __init__(self, human_policy_fn=query_response_cake_pizza):
        assistance_game = CakePizzaGraphGame()
        super().__init__(assistance_game=assistance_game, human_policy_fn=human_policy_fn)

    def render(self):
        print('s: ', self.state % 8)


class CakePizzaTimeDependentAG(AssistanceGame):
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
    

def query_response_cake_pizza_time_dep(time_before_feedback_available=0):
    def time_dep_policy_fn(assistance_game, reward, **kwargs):
        """Hardcoded query response for the time-dependent cake-pizza game. If in the querying state and able to give
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


class CakePizzaTimeDependentProblem(AssistanceProblem):
    def __init__(self, use_belief_space=True):
        human_policy_fn = query_response_cake_pizza_time_dep(time_before_feedback_available=4)
        self.assistance_game = CakePizzaTimeDependentAG(horizon=10)
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


class CakePizzaGridAG(AssistanceGame):
    State = namedtuple('State', ['pos_x', 'pos_y', 'meal', 'meal_timer', 'drink', 'query', 'time'])
    def __init__(self, spec):
        self.height = spec.height
        self.width = spec.width
        self.meal_pos = spec.meal_pos
        self.drink_pos = spec.drink_pos
        self.horizon = spec.horizon
        self.meal_cooking_time = spec.meal_cooking_time
        n_world_actions = 7
        self.n_queries = 2
        # robot actions are noop, 4 actions for moving, 2 for cooking, and two queries
        self.robot_action_space = Discrete(n_world_actions + self.n_queries)
        # human actions are no-op and the binary query response (separate queries don't get separate response actions)
        self.human_action_space = Discrete(3)

        self.enumerate_states()

        init_state = self.State(pos_y=0, pos_x=0, meal=0, meal_timer=0, drink=0, query=0, time=0)
        init_state_dist = np.zeros(self.nS)
        init_state_dist[self.get_idx(init_state)] = 1.0
        transition, rewards_dist = self.make_transition_and_reward_matrices()

        # How many different values can each state feature take. This shouldn't be interpreted as a state!
        self.discrete_feature_dims = self.State(pos_y=self.height,
                                                pos_x=self.width,
                                                meal=5,
                                                meal_timer=spec.meal_cooking_time + 1,
                                                drink=4,
                                                query=self.n_queries + 1,
                                                time=self.horizon)
        self.max_feature_value = max(self.discrete_feature_dims)
        self.one_hot_features = {'pos_x', 'pos_y', 'meal', 'drink', 'query'}
        num_regular_features = len(init_state) - len(self.one_hot_features)
        len_one_hot_features = 0
        for feature in self.one_hot_features:
            assert hasattr(init_state, feature)
            len_one_hot_features += getattr(self.discrete_feature_dims, feature)
        self.feature_vector_length = num_regular_features + len_one_hot_features

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
                              query=0,
                              time=min(s.time + 1, self.horizon))

        new_pos_x, new_pos_y, new_meal, new_meal_timer, new_drink, new_query = s.pos_x, s.pos_y, s.meal, s.meal_timer, s.drink, s.query

        # If the meal or the drink is ready, it is automatically served / transitioned into an own absorbing state.
        # This way the agent spends only one timestep in a state with pizza / drink, and is rewarded once.
        if s.meal in [2, 3] and s.meal_timer == 0:
            new_meal = 4
        if s.drink in [1, 2]:
            new_drink = 3

        # movement
        if a_r in [1, 2, 3, 4]:
            move = [(1, 0), (-1, 0), (0, -1), (0, 1)][a_r - 1]
            new_pos = (s.pos_y + move[0], s.pos_x + move[1])
            if (0 <= new_pos[0] <= self.height-1 and 0 <= new_pos[1] <= self.width-1
                    and new_pos not in [self.meal_pos, self.drink_pos]):
                new_pos_x, new_pos_y = new_pos[1], new_pos[0]

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
                elif s.meal in [2, 3]:
                    # tick down the baking timer if it's not 0
                    if s.meal_timer > 0:
                        new_meal_timer -= 1

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
                          query=new_query,
                          time=min(s.time + 1, self.horizon))

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
                            for time in range(self.horizon+1):
                                for query in range(self.n_queries + 1):
                                    s = self.State(pos_x=x,
                                                   pos_y=y,
                                                   meal=meal,
                                                   meal_timer=meal_timer,
                                                   drink=drink,
                                                   query=query,
                                                   time=time)
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
            r += 10
        elif s.meal_timer == 0 and ((s.meal == 2 and not prefer_pizza) or (s.meal == 3 and prefer_pizza)):
            r += -5

        if (s.drink == 1 and prefer_lemonade) or (s.drink == 2 and not prefer_lemonade):
            r += 10
        elif (s.drink == 1 and not prefer_lemonade) or (s.drink == 2 and prefer_lemonade):
            r += -5
        if s.time == self.horizon and (s.drink == 0 or s.meal in [0, 1]):
            r -= 10
        return r

    def get_state_features(self, s):
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
 #       return np.asarray(s, dtype='float32')


def human_response_cake_pizza_grid(time_before_feedback_available=10):
    def time_dep_policy_fn(assistance_game, reward, **kwargs):
        ag = assistance_game
        reward_idx = kwargs['reward_idx']
        prefer_pizza, prefer_lemonade = [(True, True), (True, False), (False, True), (False, False)][reward_idx]
        policy = np.zeros((ag.state_space.n, ag.human_action_space.n))
        for idx in range(ag.nS):
            s = ag.get_state(idx)
            # noop
            if s.query == 0 or s.time < time_before_feedback_available:
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
    return time_dep_policy_fn


class CakePizzaGridProblem(AssistanceProblem):
    Spec = namedtuple('Spec', ['height', 'width', 'meal_pos', 'meal_cooking_time', 'drink_pos', 'horizon', 'discount', 'time_before_feedback_available'])
    def __init__(self, use_belief_space=True):
        spec = self.Spec(height=2,
                         width=2,
                         meal_pos=(1, 1),
                         meal_cooking_time=3,
                         drink_pos=(0, 1),
                         horizon=20,
                         discount=0.98,
                         time_before_feedback_available=5)
        human_policy_fn = human_response_cake_pizza_grid(spec.time_before_feedback_available)
        self.assistance_game = CakePizzaGridAG(spec)
        ag = self.assistance_game

        if use_belief_space:
            observation_model_fn = BeliefObservationModel
        else:
            #feature_extractor = lambda state : state % self.assistance_game.state_space.n
            feature_extractor = lambda s_idx : ag.get_state_features(ag.get_state(s_idx % ag.state_space.n))

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

#        super().__init__(assistance_game=self.assistance_game, human_policy_fn=human_policy_fn)

    def render(self, mode='human'):
        s = self.assistance_game.get_state(self.state % self.assistance_game.state_space.n)
        s_str = 'pos: ({}, {}), meal: {}, meal_timer: {}, drink: {}, t: {}'.format(s.pos_y,
                                                                                   s.pos_x,
                                                                                   s.meal,
                                                                                   s.meal_timer,
                                                                                   s.drink,
                                                                                   s.time)
        if s.query:
            s_str = s_str + ' query={}'.format(s.query)
        print(s_str)




















