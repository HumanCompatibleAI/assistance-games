"""Ready-to-use POMDP and AssistanceProblem environments.
"""
import os

import gym
from gym.spaces import Discrete, MultiDiscrete
import numpy as np
import pkg_resources

from assistance_games.core import POMDP, AssistanceGame, AssistanceProblem, hard_value_iteration, query_response_cake_pizza, s_reward_to_saas_reward
from assistance_games.parser import read_pomdp
import assistance_games.rendering as rendering
from assistance_games.utils import get_asset, sample_distribution

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

    def __init__(self, terminal=False, horizon=None, sample_ob_on_reset=False):
        if terminal:
            pomdp = read_pomdp(get_asset('pomdps/four-three-terminal.pomdp'))
        else:
            pomdp = read_pomdp(get_asset('pomdps/four-three.pomdp'))
        self.__dict__ = pomdp.__dict__
        if horizon is not None:
            self.horizon = horizon
        self.sample_ob_on_reset = sample_ob_on_reset
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
    def __init__(self, human_policy_fn=hard_value_iteration):
        assistance_game = RedBlueAssistanceGame()
        super().__init__(assistance_game=assistance_game, human_policy_fn=human_policy_fn)


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

        ob = self.state % self.num_obs
        human_state = ob // 3
        robot_state = ob % 3

        human_coords = self.human_grid.coords_from_pos(human_grid_pos[human_state])
        robot_coords = self.robot_grid.coords_from_pos(robot_grid_pos[robot_state])

        self.human_transform.set_translation(*human_coords)
        self.robot_transform.set_translation(*robot_coords)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')


class CakePizzaGraphGame(AssistanceGame):
    def __init__(self):

        n_world_states = 4
        self.num_world_states = n_world_states
        n_world_actions = 2
        n_queries = 1
        self.num_queries = n_queries



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
        print('s: ',self.state % 8)
