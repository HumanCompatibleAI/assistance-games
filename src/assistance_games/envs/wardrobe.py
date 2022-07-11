from gym.spaces import Discrete, Box
import copy
import numpy as np

import assistance_games.rendering as rendering
from assistance_games.core import AssistancePOMDPWithMatrixSupport, UniformDiscreteDistribution, KroneckerDistribution
from assistance_games.envs.gridworld import Gridworld, Direction, make_image_renderer


class Wardrobe(AssistancePOMDPWithMatrixSupport):
    """

    This is the map of the assistance game:

    .......
    .     .
    .     .
    . WW  .
    . WW  .
    .H   R.
    .......

    H = Human
    R = Robot
    W = Wardrobe

    The human wants to move the wardrobe to one of the corners.
    To move the wardrobe, two agents need to push it at the same time
    in the same direction.

    Size of the map and of the wardrobe can be chosen.

    States are namedtuples where the keys H, R, and W are mapped to
    the positions of the human, robot and wardrobe respectively.
    """

    def __init__(self, size=3):
        self.size = size
        obs_space_highs = np.array(([size - 1] * 6) + [4])

        self.nS = (size ** 6)
        self.nAH = 4
        self.nAR = 4
        self.nOR = self.nS  # Fully observable
        layout = [" " * size for _ in range(size)]
        initial_state = {
            'H': (0, 0),
            'R': (self.size - 1, 0),
            'W': (1, 1),
        }
        rendering_fns = {
            'H': [make_image_renderer('images/girl1.png')],
            'R': [make_image_renderer('images/robot1.png')],
            'W': [make_image_renderer('images/wardrobe1.png')],
        }
        self.gridworld = Gridworld(layout, initial_state, rendering_fns, set(['W']))

        w_size = size
        super().__init__(
            human_policy_type={'H': 'optimal', 'R': 'optimal'},
            # For some reason PBVI is very sensitive to the discount -- at discount 0.9 it doesn't work.
            discount=0.7,
            horizon=3*size,
            theta_dist=UniformDiscreteDistribution([(0, 0), (0, w_size-1), (w_size-1, 0), (w_size-1, w_size-1)]),
            init_state_dist=KroneckerDistribution(initial_state),
            # Observation space should really be tuple of discretes
            observation_space=Box(low=np.array([0] * 7), high=obs_space_highs),
            action_space=Discrete(4),
            default_aH=0,
            default_aR=0,
            deterministic=True,
            fully_observable=True
        )

    def state_to_index(self, state):
        state_as_sequence = state['H'] + state['R'] + state['W']
        index = 0
        for val in state_as_sequence:
            index = index * self.size + val
        return index

    def index_to_state(self, num):
        state_as_sequence = []
        for _ in range(6):
            state_as_sequence.append(num % self.size)
            num = num // self.size

        return {
            'H': (state_as_sequence[5], state_as_sequence[4]),
            'R': (state_as_sequence[3], state_as_sequence[2]),
            'W': (state_as_sequence[1], state_as_sequence[0]),
        }

    def encode_obs_distribution(self, obs_dist, prev_aH):
        # Observations are deterministic, so extract it
        (obs,) = tuple(obs_dist.support())
        h, r, w = obs['H'], obs['R'], obs['W']
        return KroneckerDistribution(np.array(h + r + w + (prev_aH,)))

    def decode_obs(self, encoded_obs):
        state = {
            'H': encoded_obs[0:2],
            'R': encoded_obs[2:4],
            'W': encoded_obs[4:6],
        }
        return state, encoded_obs[6]

    def get_transition_distribution(self, state, aH, aR):
        aH_direction = Direction.get_direction_from_number(aH)
        aR_direction = Direction.get_direction_from_number(aR)
        next_human_pos = self.gridworld.get_move_location('H', aH_direction, state)
        next_robot_pos = self.gridworld.get_move_location('R', aR_direction, state)

        if (aH == aR and next_human_pos == state['W'] and next_robot_pos == state['W']):
            state, _ = self.gridworld.functional_move('W', aH_direction, state)
        state, _ = self.gridworld.functional_move('R', aR_direction, state)
        state, _ = self.gridworld.functional_move('H', aH_direction, state)
        return KroneckerDistribution(state)

    def get_reward(self, state, aH, aR, next_state, theta):
        if next_state['W'] == theta and state['W'] != theta:
            return 1.0
        return 0.0

    def is_terminal(self, state):
        return False

    def close(self):
        self.gridworld.close()
        return super().close()

    def render(self, state, prev_aH, prev_aR, theta, mode='human'):
        self.gridworld.set_object_positions(state)
        return self.gridworld.render(mode=mode, finalized=True)
