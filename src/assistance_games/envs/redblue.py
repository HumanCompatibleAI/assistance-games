from gym.spaces import Discrete, Box
import numpy as np

import assistance_games.rendering as rendering
from assistance_games.core import AssistancePOMDPWithMatrixSupport, UniformDiscreteDistribution, KroneckerDistribution
from assistance_games.envs.gridworld import Gridworld, make_image_renderer, make_cell_renderer, make_ellipse_renderer


class RedBlue(AssistancePOMDPWithMatrixSupport):
    """Red-blue problem. Fully observable assistance POMDP.

    A state / observation is a Numpy array of length 2, encoding the human's state followed by the robot's state.
    """
    def __init__(self):
        self.nS = 12  # 4 human locations, 3 robot locations
        self.nAH = 2
        self.nAR = 3
        self.nOR = self.nS  # Fully observable

        layout = [
            "XXXXXX",
            "XBXXBX",
            "X  X X",
            "XRXXRX",
            "XXXXXX",
        ][::-1] # Reverse to make the 0th index the bottom row
        player_positions = {'robot': (4, 2), 'human': (2, 2)}
        rendering_fns = {
            'human': [make_image_renderer('images/girl1.png')],
            'robot': [make_image_renderer('images/robot1.png')],
            'X': [make_cell_renderer((0.8, 0.75, 0.7))],
            'R': [make_ellipse_renderer(scale_width=0.7, rgb_color=(0.8, 0.2, 0.2))],
            'B': [make_ellipse_renderer(scale_width=0.7, rgb_color=(0.2, 0.2, 0.8))],
        }
        self.gridworld = Gridworld(layout, player_positions, rendering_fns)

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

    def encode_obs_distribution(self, obs_dist, prev_aH):
        # Observations are deterministic, so extract it
        (obs,) = tuple(obs_dist.support())
        return KroneckerDistribution(np.array(obs + [prev_aH]))

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
        self.gridworld.close()
        return super().close()

    def render(self, state, prev_aH, prev_aR, theta, mode='human'):
        if not self.gridworld.is_renderer_initialized():
            self.gridworld.initialize_renderer(viewer_bounds=(600, 500))

        human_grid_pos = [(2, 2), (1, 2), (1, 1), (1, 3)]
        robot_grid_pos = [(4, 2), (4, 1), (4, 3)]
        human_state, robot_state = state
        self.gridworld.set_object_positions({
            'human': human_grid_pos[human_state],
            'robot': robot_grid_pos[robot_state],
        })
        return self.gridworld.render(mode=mode, finalized=True)
