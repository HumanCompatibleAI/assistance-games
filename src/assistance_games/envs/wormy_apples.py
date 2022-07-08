from enum import IntEnum
from functools import partial
from gym.spaces import Discrete, Box
import numpy as np

import assistance_games.rendering as rendering
from assistance_games.utils import get_asset
from assistance_games.core import (
    AssistancePOMDPWithMatrixSupport,
    UniformDiscreteDistribution,
    DiscreteDistribution,
    KroneckerDistribution
)

class WormyApplesState(IntEnum):
    INIT_STATE = 0
    CLEAN_APPLE = 1
    WORMY_APPLE = 2
    COMPOST = 3
    TRASH = 4
    PIE = 5


class AR(IntEnum):
    NOOP = 0
    GET_APPLE = 1
    TOSS_COMPOST = 2
    TOSS_TRASH = 3
    MAKE_PIE = 4
    QUERY = 5


class AH(IntEnum):
    NOOP = 0
    COMPOST = 1
    TRASH = 2


class WormyApples(AssistancePOMDPWithMatrixSupport):
    """
    Robot must get apples and make a pie, but the apples can be wormy. In which
    case, the robot must first dispose of the wormy apples before it can make
    the pie, and the human has a preference over how the robot disposes of the
    apples.
    start ---> apples --> pie
           \-> wormy apples ---> compost --> pie
                             \-> trash --> pie

    If two phase is set to True, then the human only responds to the robot query
    in INIT_STATE.

    Initial implementation by Neel Alex, then rewritten by Rohin Shah. Due to
    questionable Git practices, the history of the file does not include Neel's
    implementation.
    """
    def __init__(self, two_phase=False, discount=0.99):
        self.two_phase = two_phase
        self.nS = 6
        self.nAH = 3
        self.nAR = 6
        self.nOR = self.nS  # Fully observable
        self.viewer = None

        super().__init__(
            discount=discount,
            horizon=4,
            theta_dist=UniformDiscreteDistribution(['compost', 'trash']),
            init_state_dist=KroneckerDistribution(WormyApplesState.INIT_STATE),
            # Observation space should really be tuple of discretes
            observation_space=Box(low=0.0, high=1.0, shape=(9,), dtype=np.float32),
            action_space=Discrete(6),
            default_aH=AH.NOOP,
            default_aR=AR.NOOP,
            deterministic=False,
            fully_observable=True
        )

    def index_to_state(self, num):
        """Convert numeric state to underlying env state."""
        return WormyApplesState(num)

    def index_to_human_action(self, num):
        return AH(num)

    def index_to_robot_action(self, num):
        return AR(num)

    def encode_obs_distribution(self, obs_dist, prev_aH):
        # Observations are deterministic, so extract it
        (obs,) = tuple(obs_dist.support())
        obs_one_hot = [0.0] * self.nOR
        obs_one_hot[obs] = 1.0
        prev_aH_one_hot = [0.0] * self.nAH
        prev_aH_one_hot[prev_aH] = 1.0
        return KroneckerDistribution(np.array(obs_one_hot + prev_aH_one_hot))

    def decode_obs(self, encoded_obs):
        obs_one_hot, prev_aH_one_hot = encoded_obs[:self.nOR], encoded_obs[self.nOR:]
        return (np.argmax(obs_one_hot), np.argmax(prev_aH_one_hot))

    def get_transition_distribution(self, state, aH, aR):
        S = WormyApplesState
        transitions = {
            (S.INIT_STATE, AR.GET_APPLE): {S.CLEAN_APPLE: 0.8, S.WORMY_APPLE: 0.2},
            (S.CLEAN_APPLE, AR.MAKE_PIE): {S.PIE: 1.0},
            (S.WORMY_APPLE, AR.TOSS_COMPOST): {S.COMPOST: 1.0},
            (S.WORMY_APPLE, AR.TOSS_TRASH): {S.TRASH: 1.0},
            (S.COMPOST, AR.MAKE_PIE): {S.PIE: 1.0},
            (S.TRASH, AR.MAKE_PIE): {S.PIE: 1.0},
        }
        trans_dict = transitions.get((state, aR), {state: 1.0})
        return DiscreteDistribution(trans_dict)

    def get_reward(self, state, aH, aR, next_state, theta):
        reward = 0.0
        if state == next_state:
            pass
        elif (next_state == WormyApplesState.COMPOST and theta != 'compost') or \
             (next_state == WormyApplesState.TRASH and theta != 'trash'):
            reward -= 2.0
        elif next_state == WormyApplesState.PIE:
            reward += 2.0
        if aR == AR.QUERY:
            reward -= 0.1
        return reward

    def get_human_action_distribution(self, obsH, prev_aR, theta):
        aH = AH.NOOP
        if prev_aR == AR.QUERY and not (self.two_phase and obsH != WormyApplesState.INIT_STATE):
            aH = AH.COMPOST if theta == 'compost' else AH.TRASH
        return KroneckerDistribution(aH)

    def is_terminal(self, state):
        return False

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
        return super().close()

    def state_string(self, s):
        descriptions = [
            "Start",
            "Got apples, no worms",
            "Got apples, there are worms",
            "Apples, worms in compost",
            "Apples, worms in trash",
            "Pie made"
        ]
        return descriptions[s]

    def render(self, state, prev_aH, prev_aR, theta, mode="human"):
        # Render of the initial state will set previous actions to None
        if prev_aR is not None:
            print('Robot: {}'.format(prev_aR.name))
            print('Human: {}'.format(prev_aH.name))
        print('State: {}'.format(self.state_string(state)))
        # return
        width = 5
        height = 3

        grid_side = 30
        gs = grid_side

        def make_image_transform(filename, w=1.0, h=None, s=0.6, c=None):
            if h is None: h = w
            fullname = get_asset(f'images/{filename}')
            img = rendering.Image(fullname, s * w * gs, s * h * gs)
            transform = rendering.Transform()
            img.add_attr(transform)

            if c is not None:
                img.set_color(*c)

            return img, transform

        make_item_image = {
            'apples': partial(make_image_transform, 'apples.png'),
            'apple': partial(make_image_transform, 'apple_green.png'),
            'wormy_apple': partial(make_image_transform, 'wormy_apple.png'),
            'trash': partial(make_image_transform, 'trash.png'),
            'compost': partial(make_image_transform, 'compost.png'),
            'pie': partial(make_image_transform, 'apple-pie1.png'),
            'oven': partial(make_image_transform, 'plate1.png'),
            'question': partial(make_image_transform, 'question_mark.png'),
        }

        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 800)
            self.viewer.set_bounds(-130, 120, -150, 250)

            g_x0 = -110 + grid_side
            g_y0 = -110 + grid_side

            self.grid = rendering.Grid(start=(g_x0, g_y0), grid_side=grid_side, shape=(width, height))
            self.grid.set_color(0.85, 0.85, 0.85)
            self.viewer.add_geom(self.grid)

            human_image = get_asset('images/girl1.png') # TODO
            human = rendering.Image(human_image, grid_side, grid_side)
            self.human_transform = rendering.Transform()
            human.add_attr(self.human_transform)
            self.viewer.add_geom(human)

            robot_image = get_asset('images/robot1.png') # TODO
            robot = rendering.Image(robot_image, grid_side, grid_side)
            self.robot_transform = rendering.Transform()
            robot.add_attr(self.robot_transform)
            self.viewer.add_geom(robot)

            ### Render counters

            make_rect = lambda x, y, w, h: rendering.make_polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])

            make_grid_rect = lambda i, j, di, dj: make_rect(g_x0 + i * gs, g_y0 + j * gs, di * gs, dj * gs)

            counters = [
                make_grid_rect(-1, -1, width + 2, 1),
                make_grid_rect(-1, -1, 1, height + 2),

                make_grid_rect(-1, height, width + 2, 1),
                make_grid_rect(width, -1, 1, height + 2),

                make_grid_rect(width - 2, -1, 1, height + 2),
            ]

            for counter in counters:
                r = 0.8
                off = 0.05
                g = r - off
                b = r - 2 * off
                counter.set_color(r, g, b)
                self.viewer.add_geom(counter)

            item_positions = {
                'apples': (-1, 1),
                'trash': (1, -1),
                'compost': (1, height),
                'oven': (3, 1)
            }
            for itemname, position in item_positions.items():
                coords = self.grid.coords_from_pos(position)

                item, transform = make_item_image[itemname]()
                transform.set_translation(*coords)
                self.viewer.add_geom(item)

            ### Render formulae
            # Sort this out later
            if False:
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
                        tr.set_translation(hl * j, 1.2 * hl * i)
                        self.viewer.add_geom(img)

            # Render thought bubble

            scale = 0.8
            thought = rendering.make_ellipse(scale * grid_side / 2, scale * 0.7 * grid_side / 2)
            thought.set_color(0.9, 0.9, 0.9)
            thought_transform = rendering.Transform()
            thought_transform.set_translation(-gs, gs)
            thought.add_attr(thought_transform)
            thought.add_attr(self.human_transform)

            self.viewer.add_geom(thought)

            scale = 0.17
            thought2 = rendering.make_ellipse(scale * grid_side / 2, scale * grid_side / 2)
            thought2.set_color(0.9, 0.9, 0.9)
            thought_transform2 = rendering.Transform()
            thought_transform2.set_translation(-0.6 * gs, 0.6 * gs)
            thought2.add_attr(thought_transform2)
            thought2.add_attr(self.human_transform)

            self.viewer.add_geom(thought2)

            scale = 0.1
            thought3 = rendering.make_ellipse(scale * grid_side / 2, scale * grid_side / 2)
            thought3.set_color(0.9, 0.9, 0.9)
            thought_transform3 = rendering.Transform()
            thought_transform3.set_translation(-0.4 * gs, 0.4 * gs)
            thought3.add_attr(thought_transform3)
            thought3.add_attr(self.human_transform)

            self.viewer.add_geom(thought3)


            self.bins = []
            for bin_name in ('trash', 'compost'):
                bin, _ = make_item_image[bin_name](s=0.4)
                bin.add_attr(self.human_transform)
                bin.add_attr(thought_transform)
                self.bins.append(bin)

        human_pos = (4, 1)
        plate_pos = (3, 1)
        robot_attributes = [
            ((2, 1), ''),
            ((0, 1), 'apple'),
            ((0, 1), 'wormy_apple'),
            ((1, 0), ''),
            ((1, 2), ''),
            ((2, 1), ''),
        ]
        robot_pos, robot_hand = robot_attributes[state]

        human_coords = self.grid.coords_from_pos(human_pos)
        self.human_transform.set_translation(*human_coords)

        robot_coords = self.grid.coords_from_pos(robot_pos)
        self.robot_transform.set_translation(*robot_coords)

        preferred_idx = 0 if theta == 'compost' else 1
        question = prev_aR == AR.QUERY
        ### Render hand content

        for hand, hand_transform in (
                # (human_hand, self.human_transform),
                (robot_hand, self.robot_transform),
        ):
            if hand != '':
                item, transform = make_item_image[hand](s=0.4)
                transform.set_translation(0, -5)
                item.add_attr(hand_transform)
                self.viewer.add_onetime(item)

        # items_to_pos = {item: pos for pos, item in self.ag.counter_items.items()}
        # plate_pos = move_to_counter(items_to_pos['P'])
        # plate_coords = self.grid.coords_from_pos(plate_pos)

        ### Render question asked by robot
        if question:
            # TODO position question s.t. it never overlaps with human though bubble
            scale = 0.8
            speech = rendering.make_ellipse(scale * grid_side / 2, scale * 0.7 * grid_side / 2)
            speech.set_color(0.9, 0.9, 0.9)
            speech_transform = rendering.Transform()
            speech_transform.set_translation(gs, gs)
            speech.add_attr(speech_transform)
            speech.add_attr(self.robot_transform)

            self.viewer.add_onetime(speech)

            # triangle base
            # speech_triangle = rendering.make_triangle()
            # speech_triangle.set_color(0.9, 0.9, 0.9)
            # speech_triangle_transform = rendering.Transform()
            # speech_triangle_transform.set_translation(gs, gs)
            # speech_triangle.add_attr(speech_triangle_transform)
            # speech_triangle.add_attr(self.robot_transform)
            #
            # self.viewer.add_onetime(speech_triangle)

            trash_q, transform = make_item_image['trash'](s=0.3)
            transform.set_translation(gs - 5, gs)
            trash_q.add_attr(self.robot_transform)
            self.viewer.add_onetime(trash_q)

            q_mark, transform = make_item_image['question'](s=0.3)
            transform.set_translation(gs + 5, gs)
            q_mark.add_attr(self.robot_transform)
            self.viewer.add_onetime(q_mark)
        # TODO add human response
        ### Render plate content
        if self.state_string(state) == "Pie made":
            pie, transform = make_item_image['pie'](s=0.4)
            transform.set_translation(*self.grid.coords_from_pos(plate_pos))
            self.viewer.add_onetime(pie)

        ### Render pie in thought bubble
        self.viewer.add_onetime(self.bins[preferred_idx])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
