from gym.spaces import Discrete, Box
import numpy as np

import assistance_games.rendering as rendering
from assistance_games.utils import get_asset
from assistance_games.core.core2 import AssistancePOMDPWithMatrixSupport, UniformDiscreteDistribution, KroneckerDistribution

class RedBlue2(AssistancePOMDPWithMatrixSupport):
    """Red-blue problem. Fully observable assistance POMDP.

    A state / observation is a Numpy array of length 2, encoding the human's state followed by the robot's state.
    """
    def __init__(self):
        self.nS = 12  # 4 human locations, 3 robot locations
        self.nAH = 2
        self.nAR = 3
        self.nOR = self.nS  # Fully observable
        self.viewer = None

        super().__init__(
            discount=0.9,
            horizon=4,
            theta_dist=UniformDiscreteDistribution(['red', 'blue']),
            init_state_dist=KroneckerDistribution([0, 0]),
            # Observation space should really be tuple of discretes
            observation_space=Box(low=np.array([0, 0, 0]), high=np.array([3, 2, 1])),
            action_space=Discrete(3),
            default_aH=0,
            default_aR=0,
            deterministic=True,
            fully_observable=True
        )

    def state_to_index(self, state):
        h, r = state
        return h * 3 + r

    def index_to_state(self, num):
        return (num // 3), (num % 3)

    def encode_obs(self, obs, prev_aH):
        return np.array(obs + [prev_aH])

    def decode_obs(self, encoded_obs):
        h, r, prev_aH = encoded_obs
        return (h, r), prev_aH

    def get_transition_distribution(self, state, aH, aR):
        h, r = state
        if h == 0:
            newH = 1
        elif h >= 2:
            newH = h
        else:
            newH = aH + 2

        if r >= 1:
            newR = r
        else:
            newR = (aR + 1) % 3
        return KroneckerDistribution([newH, newR])

    def get_reward(self, state, aH, aR, next_state, theta):
        h, r = state
        reward = 0.0
        if h == 1 and ((theta == 'red' and aH == 0) or (theta == 'blue' and aH == 1)):
            reward += 1.0
        if r == 0 and ((theta == 'red' and aR == 0) or (theta == 'blue' and aR == 1)):
            reward += 1.0
        return reward

    def get_human_action_distribution(self, obsH, prev_aR, theta):
        h, r = obsH
        aH = 0
        if h == 1 and theta == 'blue':
            aH = 1
        return KroneckerDistribution(aH)

    def is_terminal(self, state):
        return False

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
        return super().close()

    def render(self, state, theta, mode='human'):
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

        human_state, robot_state = state

        human_coords = self.human_grid.coords_from_pos(human_grid_pos[human_state])
        robot_coords = self.robot_grid.coords_from_pos(robot_grid_pos[robot_state])

        self.human_transform.set_translation(*human_coords)
        self.robot_transform.set_translation(*robot_coords)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
