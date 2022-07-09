from copy import deepcopy
from functools import partial
from gym.spaces import Discrete, Box
import numpy as np

import assistance_games.rendering as rendering
from assistance_games.utils import get_asset
from assistance_games.core import AssistancePOMDP, UniformContinuousDistribution, KroneckerDistribution
from assistance_games.envs.gridworld import Gridworld, make_image_renderer, make_cell_renderer, make_ellipse_renderer, Direction


class SmallPieGridworld(AssistancePOMDP):
    def __init__(self):
        layout = [
            "XXXXXXX",
            "X     X",
            "X     P",
            "X     X",
            "XXCBAXX",
        ][::-1] # Reverse so that index 0 means what is visually the bottom row
        player_positions = {'R': (1, 1), 'H': (1, 3)}
        rendering_fns = {
            'H': [make_ellipse_renderer(scale_width=0.8, scale_height=0.56, offset=(-1, 1), rgb_color=(0.9, 0.9, 0.9)),
                  make_ellipse_renderer(scale_width=0.17, scale_height=0.12, offset=(-0.6, 0.6), rgb_color=(0.9, 0.9, 0.9)),
                  make_ellipse_renderer(scale_width=0.1, scale_height=0.07, offset=(-0.4, 0.4), rgb_color=(0.9, 0.9, 0.9)),
                  make_image_renderer('images/girl1.png')],
            'R': [make_image_renderer('images/robot1.png')],
            'X': [make_cell_renderer((0.8, 0.75, 0.7))],
            'A': [make_cell_renderer((0.8, 0.75, 0.7)),
                  make_image_renderer('images/apple3.png', scale=0.9, rgb_color=(0.7, 0.3, 0.2))],
            'B': [make_cell_renderer((0.8, 0.75, 0.7)),
                  make_image_renderer('images/apple3.png', scale=0.9, rgb_color=(0.5, 0.7, 0.0))],
            'C': [make_cell_renderer((0.8, 0.75, 0.7)),
                  make_image_renderer('images/chocolate2.png', scale=0.9)],
            'P': [make_cell_renderer((0.8, 0.75, 0.7)),
                  make_image_renderer('images/plate1.png')],
        }
        self.gridworld = Gridworld(layout, player_positions, rendering_fns)

        initial_state = {
            'pos' : (1, 1),
            'orientation' : Direction.NORTH,
            'hand' : '',
            'plate' : (),
            'pie' : None,
        }

        self.recipes = [
            ('A',),
            ('B',),
            ('C',)
        ]
        self.ingredients = ['A', 'B', 'C']
        self.ingredient_to_index = {v: i for i, v in enumerate(self.ingredients)}
        # position, hand + plate, human response
        num_dims = (self.gridworld.width-2) + (self.gridworld.height-2) + (2*len(self.ingredients)) + 1
        low = np.zeros(num_dims)
        high = np.ones(num_dims)
        high[-1] = 3.0
        super().__init__(
            discount=0.99,
            horizon=20,
            theta_dist=UniformContinuousDistribution([1, 0, 0], [3, 2, 2]),
            init_state_dist=KroneckerDistribution(initial_state),
            observation_space=Box(low=low, high=high),
            action_space=Discrete(9),
            default_aH=0,
            default_aR=0
        )
        self.INTERACT = 5

    def get_transition_distribution(self, state, aH, aR):
        s = deepcopy(state)
        obj_pos = {'H': (1, 3), 'R': state['pos']}
        obj_or = {'H': Direction.NORTH, 'R': state['orientation']}

        if aR == self.INTERACT:
            wall_type = self.gridworld.get_facing_wall_type('R', obj_pos, obj_or)
            s['hand'] = self._interact_hand(state, wall_type)
            s['plate'] = self._interact_plate(state, wall_type)
            s['pie'] = self._interact_pie(state, wall_type)
        else:
            aR_direction = Direction.get_direction_from_number(aR)
            gridworld_positions, gridworld_orientations = self.gridworld.functional_move(
                'R', aR_direction, obj_pos, obj_or)
            s['pos'] = gridworld_positions['R']
            s['orientation'] = gridworld_orientations['R']

        return KroneckerDistribution(s)

    def _interact_hand(self, state, wall_type):
        # Can pick up an item
        if wall_type in self.ingredients and state['hand'] == '':
            return wall_type
        # Can put an item on a plate or back in its location
        elif wall_type in (state['hand'], 'P'):
            return ''
        return state['hand']

    def _interact_plate(self, state, wall_type):
        if state['hand'] != '' and wall_type == 'P' and len(state['plate']) < 4:
            return tuple(sorted(state['plate'] + (state['hand'],)))
        return state['plate']

    def _interact_pie(self, state, wall_type):
        if state['hand'] == '' and wall_type == 'P' and state['plate'] in self.recipes:
            return state['plate']
        return None

    def get_reward(self, state, aH, aR, next_state, theta):
        if next_state['pie'] is None:
            return 0

        preferred_idx = max(range(len(theta)), key=lambda i: theta[i]) # argmax
        actual_idx = [i for i, recipe in enumerate(self.recipes) if recipe == next_state['pie']][0]
        return 10 if actual_idx == preferred_idx else 1

    def get_human_action_distribution(self, obsH, prev_aR, theta):
        question = prev_aR - 6
        answer = theta[question] if question >=0 else 0
        return KroneckerDistribution(answer)

    def is_terminal(self, state):
        return state['pie'] != None

    def encode_obs_distribution(self, obs_dist, prev_aH):
        # Observations are deterministic, so extract it
        (obs,) = tuple(obs_dist.support())
        num_ingredients = len(self.ingredients)
        def one_hot(i, n):
            result = np.zeros(n)
            result[i] = 1.0
            return result

        x, y = obs['pos']
        # Chop off 1 on both sides to ignore counters
        x_ob = one_hot(x-1, self.gridworld.width-2)
        y_ob = one_hot(y-1, self.gridworld.height-2)

        hand_ob = np.zeros(len(self.ingredients))
        if obs['hand'] != '':
            hand_ob[self.ingredient_to_index[obs['hand']]] = 1.0

        plate_ob = np.zeros(len(self.ingredients))
        for item in obs['plate']:
            plate_ob[self.ingredient_to_index[item]] = 1.0

        aH_ob = np.array([prev_aH])

        full_ob = np.concatenate([x_ob, y_ob, hand_ob, plate_ob, aH_ob])
        return KroneckerDistribution(full_ob)

    def close(self):
        self.gridworld.close()
        super().close()

    def render(self, state, prev_aH, prev_aR, theta, mode='human'):
        gw, gh = 200.0 / self.gridworld.width, 200.0 / self.gridworld.height

        def make_image_transform(filename, w=1.0, h=None, s=0.6, c=None):
            if h is None: h = w
            fullname = get_asset(f'images/{filename}')
            img = rendering.Image(fullname, s * w * gw, s * h * gh)
            transform = rendering.Transform()
            img.add_attr(transform)

            if c is not None:
                img.set_color(*c)

            return img, transform

        make_item_image = {
            'A' : partial(make_image_transform, 'apple3.png', c=(0.7, 0.3, 0.2)),
            'B' : partial(make_image_transform, 'apple3.png', c=(0.5, 0.7, 0.0)),
            'C' : partial(make_image_transform, 'chocolate2.png'),
            'F' : partial(make_image_transform, 'flour3.png'),
            'P' : partial(make_image_transform, 'plate1.png', w=1.3, h=1.3),
            '+' : partial(make_image_transform, 'plus1.png', w=0.5, h=0.5),
            '=' : partial(make_image_transform, 'equal1.png', w=0.5, h=0.2),
            '2' : partial(make_image_transform, 'apple-pie1.png'),
            '1' : partial(make_image_transform, 'apple-pie1.png', c=(0.5, 0.7, 0.0)),
            '0' : partial(make_image_transform, 'apple-pie1.png', c=(0.7, 0.3, 0.2)),
        }

        self.gridworld.set_object_positions({'R': state['pos'], 'H': (1, 3)})

        if not self.gridworld.is_renderer_initialized():
            self.gridworld.initialize_renderer(viewer_bounds=(600, 800), grid_offsets=(0, 0, 0, 130))
            self.human_transform = rendering.Transform()
            self.robot_transform = rendering.Transform()

            ### Render formulae
            header_x = -120 + gw
            header_y = -150 + 8 * gh
            hl = 20

            header_transform = rendering.Transform()
            header_transform.set_translation(header_x, header_y)

            for i, recipe in enumerate(self.recipes):
                formula = '+'.join(recipe) + f'={i}'
                for j, c in enumerate(formula):
                    img, tr = make_item_image[c](s=0.5)
                    img.add_attr(header_transform)
                    tr.set_translation(hl*j, 1.2*hl*i)
                    self.gridworld.viewer.add_geom(img)

            ### Pie objects for thought bubble
            self.pies = []
            for pie_idx in ('012'):
                pie, pie_transform = make_item_image[pie_idx](s=0.4)
                pie.add_attr(self.human_transform)
                pie_transform.set_translation(-gw, gh)
                self.pies.append(pie)

        self.gridworld.render(mode=mode)

        human_coords = self.gridworld.grid.coords_from_pos((1, 3))
        self.human_transform.set_translation(*human_coords)
        robot_coords = self.gridworld.grid.coords_from_pos(state['pos'])
        self.robot_transform.set_translation(*robot_coords)

        ### Render pie in thought bubble
        preferred_idx = max(range(len(theta)), key=lambda i: theta[i]) # argmax
        self.gridworld.viewer.add_onetime(self.pies[preferred_idx])

        
        ### Render hand content
        if state['hand'] != '':
            item, transform = make_item_image[state['hand']](s=0.4)
            transform.set_translation(0, -5)
            item.add_attr(self.robot_transform)
            self.gridworld.viewer.add_onetime(item)

        ### Render plate content
        plate_coords = self.gridworld.grid.coords_from_pos((6, 2))
        if state['pie'] is not None:
            for idx, recipe in enumerate(self.recipes):
                if state['pie'] == recipe:
                    pie, transform = make_item_image[str(idx)](s=0.65)
                    transform.set_translation(*plate_coords)
                    self.gridworld.viewer.add_onetime(pie)
                    recipe_made = True
                    break
        else:
            for j, item_name in enumerate(state['plate']):
                item, transform = make_item_image[item_name](s=0.4)

                d = 7
                dx = (-1) ** (j+1) * d
                dy = (-1) ** (j // 2) * d
                item_coords = (lambda x, y : (x+dx, y+dy))(*plate_coords)

                transform.set_translation(*item_coords)
                self.gridworld.viewer.add_onetime(item)

        return self.gridworld.viewer.render(return_rgb_array = mode=='rgb_array')


def get_small_pie_hardcoded_robot_policy(*args, **kwargs):
    class Policy:
        def predict(self, ob, state=None):
            N, S, E, W = Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST
            N, S, E, W = map(Direction.get_number_from_direction, [N, S, E, W])
            INTERACT = 5

            # Hacky way of detecting a reset
            if state is None:
                self.actions = [E, E, E, S, INTERACT, N, E, INTERACT, INTERACT]

            if len(self.actions) == 0:
                return Direction.STAY, 'ignored'

            aR = self.actions.pop(0)
            return aR, 'ignored'

    return Policy()
