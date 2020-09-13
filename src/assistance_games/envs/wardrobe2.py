from collections import namedtuple
from gym.spaces import Discrete, Box
import numpy as np

import assistance_games.rendering as rendering
from assistance_games.utils import get_asset, MOVEMENT_ACTIONS
from assistance_games.core.core2 import AssistancePOMDPWithMatrixSupport, UniformDiscreteDistribution, KroneckerDistribution


WardrobeState = namedtuple('WardrobeState', ['H', 'R', 'W'])


class Wardrobe2(AssistancePOMDPWithMatrixSupport):
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

    States are namedtuples where the keys H, R, and W are mapped to
    the positions of the human, robot and wardrobe respectively.
    """

    def __init__(self, size=3, wardrobe_size=1):
        self.size = size
        self.wardrobe_size = wardrobe_size
        w_size = self.size - self.wardrobe_size + 1
        self.wardrobe_range = w_size
        self.num_vals_for_sequence = [size] * 4 + [w_size] * 2
        obs_space_highs = np.array(self.num_vals_for_sequence + [5]) - 1

        self.nS = (size ** 4) * (w_size ** 2)
        self.nAH = 4
        self.nAR = 4
        self.nOR = self.nS  # Fully observable
        self.viewer = None

        super().__init__(
            human_policy_type={'H': 'optimal', 'R': 'optimal'},
            discount=0.7,
            horizon=3*size,
            theta_dist=UniformDiscreteDistribution([(0, 0), (0, w_size-1), (w_size-1, 0), (w_size-1, w_size-1)]),
            init_state_dist=KroneckerDistribution(WardrobeState(
                H=(0, 0),
                R=(self.size - 1, 0),
                W=(1, 1),
            )),
            # Observation space should really be tuple of discretes
            observation_space=Box(low=np.array([0] * 7), high=obs_space_highs),
            action_space=Discrete(4),
            default_aH=0,
            default_aR=0,
            deterministic=True,
            fully_observable=True
        )

    def state_to_index(self, state):
        h, r, w = state.H, state.R, state.W
        state_as_sequence = h + r + w

        hr_size, w_size = self.size, self.wardrobe_range
        num_possible_vals = [hr_size] * 4 + [w_size] * 2

        index = 0
        for val, num_possible_val in zip(state_as_sequence, num_possible_vals):
            index = index * num_possible_val + val
        return index

    def index_to_state(self, num):
        hr_size, w_size = self.size, self.wardrobe_range
        num_possible_vals = [w_size] * 2 + [hr_size] * 4

        state_as_sequence = []
        for num_possible_val in num_possible_vals:
            state_as_sequence.append(num % num_possible_val)
            num = num // num_possible_val

        return WardrobeState(
            H=(state_as_sequence[5], state_as_sequence[4]),
            R=(state_as_sequence[3], state_as_sequence[2]),
            W=(state_as_sequence[1], state_as_sequence[0]),
        )

    def encode_obs(self, obs, prev_aH):
        h, r, w = obs.H, obs.R, obs.W
        return np.array(h + r + w + (prev_aH,))

    def decode_obs(self, encoded_obs):
        state = WardrobeState(
            H=encoded_obs[0:2],
            R=encoded_obs[2:4],
            W=encoded_obs[4:6],
        )
        return state, encoded_obs[6]

    def get_transition_distribution(self, state, aH, aR):
        next_human_pos = self._move(state.H, aH)
        next_robot_pos = self._move(state.R, aR)

        if (aH == aR and
            self._wardrobe_can_move(state.W, aH) and
            self._in_wardrobe(state, next_human_pos) and
            self._in_wardrobe(state, next_robot_pos)
        ):
            next_wardrobe_pos = self._move(state.W, aH)
        else:
            next_wardrobe_pos = state.W
            next_human_pos = next_human_pos if not self._in_wardrobe(state, next_human_pos) else state.H
            next_robot_pos = next_robot_pos if not self._in_wardrobe(state, next_robot_pos) else state.R

        state = WardrobeState(
            H=next_human_pos,
            R=next_robot_pos,
            W=next_wardrobe_pos,
        )
        return KroneckerDistribution(state)

    def get_reward(self, state, aH, aR, next_state, theta):
        if next_state.W == theta and state.W != theta:
            return 1.0
        return 0.0

    def is_terminal(self, state):
        return False

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
        return super().close()

    def _move(self, pos, act, size=None):
        assert act != 4, "STAY action not allowed"
        if size is None:
            size = self.size

        x, y = pos
        dx, dy = MOVEMENT_ACTIONS[act]

        new_x = np.clip(x + dx, 0, size - 1)
        new_y = np.clip(y + dy, 0, size - 1)

        return new_x, new_y

    def _wardrobe_can_move(self, pos, act):
        return self._move(pos, act, size=self.wardrobe_range) != pos


    def _in_wardrobe(self, state, pos):
        wardrobe_x, wardrobe_y = state.W
        x, y = pos
        return (0 <= x - wardrobe_x < self.wardrobe_size and
                0 <= y - wardrobe_y < self.wardrobe_size)

    def render(self, state, prev_aH, prev_aR, theta, mode='human'):
        size = self.size
        wardrobe_size = self.wardrobe_size

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

        human_pos, robot_pos, wardrobe_top_left = state.H, state.R, state.W

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

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
