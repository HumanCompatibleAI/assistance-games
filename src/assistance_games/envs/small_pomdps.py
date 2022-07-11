"""Some small POMDP environments.
"""
from gym.spaces import Discrete
import numpy as np

from assistance_games.core import (
    POMDPWithMatrices, UniformDiscreteDistribution
)

from assistance_games.parser import read_pomdp
from assistance_games.utils import get_asset


class TwoStatePOMDP(POMDPWithMatrices):
    """Russell and Norvig's two-state POMDP.

    There is a reward of +1 for staying in the second state,
    but the agents' observations are noisy, so they start unsure
    of where they are.

    There is a 0.3 chance of getting the wrong state as observation,
    and a 0.1 chance of moving to the wrong state.
    """
    def __init__(self, obs_noise=0.3, act_noise=0.1, horizon=4):
        O = np.array(
            [[1.0 - obs_noise, obs_noise      ],
             [obs_noise      , 1.0 - obs_noise]]
        )

        T = np.zeros((2, 2, 2))
        T[:, 0, :] = (1 - act_noise) * np.eye(2) + act_noise * np.eye(2)[[1, 0]]
        T[:, 1, :] = act_noise * np.eye(2) + (1 - act_noise) * np.eye(2)[[1, 0]]

        R = np.zeros((2, 2, 2))
        R[1, :, :] = 1
        
        init_state_dist = UniformDiscreteDistribution([0, 1])
        super().__init__(T, R, O, discount=0.9, horizon=horizon, init_state_dist=init_state_dist)


class FourThreeMaze(POMDPWithMatrices):
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

    def __init__(self, *args, terminal=False, horizon=20, **kwargs):
        if terminal:
            T, R, O, discount, dist = read_pomdp(get_asset('pomdps/four-three-terminal.pomdp'))
        else:
            T, R, O, discount, dist = read_pomdp(get_asset('pomdps/four-three.pomdp'))

        super().__init__(T, R, O, discount=discount, horizon=horizon, init_state_dist=dist)
        self.viewer = None

    def render(self, mode='human'):
        import assistance_games.rendering as rendering

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

        # for state, ratio in enumerate(self.belief):
        #     add_bar(state, ratio)
        
        self.agent_transform.set_translation(*self.coords_from_state(self.state))

        self.viewer.render(return_rgb_array = mode=='rgb_array')
