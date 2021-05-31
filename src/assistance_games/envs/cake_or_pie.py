from collections import namedtuple
from copy import deepcopy
from functools import partial
from gym.spaces import Discrete, Box
import numpy as np

import assistance_games.rendering as rendering
from assistance_games.core import AssistancePOMDP, UniformDiscreteDistribution, KroneckerDistribution
from assistance_games.envs.gridworld import Gridworld, make_image_renderer, make_cell_renderer, make_ellipse_renderer, Direction
from assistance_games.utils import get_asset


class AbstractRecipeGridworld(AssistancePOMDP):
    """Kitchen gridworld in which H can only communicate via physical actions.

    See RecipeGridworld below for an example of how to use this class.
    """
    EMPTY_HAND = ''

    def __init__(self, gridworld, initial_state, ingredients, recipes, recipe_distribution):
        initial_state['T'] = 0
        self.gridworld = gridworld
        self.ingredients = ingredients
        self.ingredient_to_index = {v: i for i, v in enumerate(self.ingredients)}
        self.recipes = [tuple(sorted(recipe)) for recipe in recipes]
        assert all(self._is_unique(r) for r in self.recipes), "Recipes should not have duplicates"
        w, h = self.gridworld.width, self.gridworld.height
        stay = Direction.get_number_from_direction(Direction.STAY)
        super().__init__(
            discount=0.99,
            horizon=20,
            theta_dist=recipe_distribution,
            init_state_dist=KroneckerDistribution(initial_state),
            # One channel each for 1) H position + orientation, 2) R pos + or,
            # 3) H hand, 4) R hand, 5) plate contents. All effectively one hot.
            # observation_space=Box(low=0, high=1, shape=((5, h, w))),
            observation_space=Box(low=0, high=1, shape=((82,))),
            action_space=Discrete(6),
            default_aH=stay,
            default_aR=stay
        )
        self.INTERACT = 5

    def get_transition_distribution(self, state, aH, aR):
        s = deepcopy(state)
        obj_pos = {'H': state['H']['pos'], 'R': state['R']['pos']}
        obj_or = {'H': state['H']['or'], 'R': state['R']['or']}

        def handle_action(player_name, player_state, action):
            new_player_state = deepcopy(player_state)
            if action == self.INTERACT:
                wall_type = self.gridworld.get_facing_wall_type(player_name, obj_pos, obj_or)
                new_player_state['hand'] = self._interact_hand(player_state, wall_type, s)
                s['plate'] = self._interact_plate(player_state, wall_type, s)
                if player_name == 'R':
                    s['recipe'] = self._interact_recipe(player_state, wall_type, s)
            else:
                direction = Direction.get_direction_from_number(action)
                gridworld_positions, gridworld_orientations = self.gridworld.functional_move(
                    player_name, direction, obj_pos, obj_or)
                new_player_state['pos'] = gridworld_positions[player_name]
                new_player_state['or'] = gridworld_orientations[player_name]
            return new_player_state

        s['R'] = handle_action('R', s['R'], aR)
        s['H'] = handle_action('H', s['H'], aH)
        s['T'] += 1
        return KroneckerDistribution(s)

    def _interact_hand(self, player_state, wall_type, state):
        # Can pick up an item
        if wall_type in self.ingredients and player_state['hand'] == self.EMPTY_HAND:
            return wall_type
        # Can put an item on a plate or back in its location
        elif wall_type in (player_state['hand'], 'P'):
            return self.EMPTY_HAND
        return player_state['hand']

    def _interact_plate(self, player_state, wall_type, state):
        if player_state['hand'] != self.EMPTY_HAND and wall_type == 'P' and len(state['plate']) < 4:
            return tuple(sorted(state['plate'] + (player_state['hand'],)))
        return state['plate']

    def _interact_recipe(self, player_state, wall_type, state):
        if player_state['hand'] == self.EMPTY_HAND and wall_type == 'P' and state['plate'] in self.recipes:
            return state['plate']
        return None

    def get_reward(self, state, aH, aR, next_state, theta):
        if self.is_terminal(next_state):
            # Pretend like you take another action that goes to an absorbing
            # state with phi = 0 that you stay in forever. This ends up
            # effectively cancelling out the contribution of next_state.
            shaping = - self.phi(state, theta)
        else:
            shaping = self.discount * self.phi(next_state, theta) - self.phi(state, theta)

        base_reward = 0
        if next_state['recipe'] == self.recipes[theta]:
            base_reward = 2
        elif next_state['recipe'] is not None:
            base_reward = -1
        return base_reward - state['T'] * 0.01 + shaping

    # Potential shaping. Finite horizon means the reward shaping
    # theorem does not apply, so the scale needs to be kept below the
    # possible rewards (so below 1).
    def phi(self, state, theta):
        if state['recipe'] is not None:
            return 0
        if not self._is_unique(state['plate']):
            return 0

        goal_recipe = set(self.recipes[theta])
        if not set(state['plate']).issubset(goal_recipe):
            return 0
        ingredient_bonuses = len(state['plate'])  # +1 for each correct plate ingredient
        ingredients_accounted_for = list(deepcopy(state['plate']))
        h_ingredient, r_ingredient = state['H']['hand'], state['R']['hand']
        # Ensure these ingredients are useful and unique
        def is_bad(hand):
            return (hand != self.EMPTY_HAND) and (hand in state['plate'] or hand not in goal_recipe)
        if is_bad(h_ingredient) or is_bad(r_ingredient) or \
           (r_ingredient != self.EMPTY_HAND and h_ingredient == r_ingredient):
            return 0

        if h_ingredient != self.EMPTY_HAND: ingredient_bonuses += 0.5
        if r_ingredient != self.EMPTY_HAND: ingredient_bonuses += 0.5
        return ingredient_bonuses * 1.0 / (len(goal_recipe) + 1)

    def _is_unique(self, lst):
        return len(set(lst)) == len(lst)

    def get_human_action_distribution(self, obsH, prev_aR, theta):
        raise NotImplementedError("Human policy must be implemented by subclass")

    def is_terminal(self, state):
        return state['recipe'] is not None or state['T'] >= self.horizon

    def encode_obs_distribution(self, obs_dist, prev_aH):
        # Observations are deterministic, so extract it
        (obs,) = tuple(obs_dist.support())
        w, h = self.gridworld.width, self.gridworld.height

        def encode_pos(pos):
            x, y = pos
            result = np.zeros((h, w))
            result[y][x] = 1.0
            return result.flatten()

        def encode_or(orientation):
            result = np.zeros(5)
            or_idx = Direction.get_number_from_direction(orientation)
            result[or_idx] = 1.0
            return result

        def encode_ingredient_list(ingredients):
            result = np.zeros(len(self.ingredients))
            for ingredient in ingredients:
                i = self.ingredient_to_index[ingredient]
                result[i] = 1.0
            return result

        def encode_hand(hand):
            lst = [] if hand == self.EMPTY_HAND else [hand]
            return encode_ingredient_list(lst)

        features = [encode_pos(obs['R']['pos']),
                    encode_pos(obs['H']['pos']),
                    encode_or(obs['H']['or']),
                    encode_or(obs['R']['or']),
                    encode_hand(obs['R']['hand']),
                    encode_hand(obs['H']['hand']),
                    encode_ingredient_list(obs['plate'])]
        # Human action is mostly inferrable from the position + orientation, so
        # we ignore it
        return KroneckerDistribution(np.concatenate(features))

    def close(self):
        self.gridworld.close()
        super().close()

    def make_item_image_transform(self, item, scale=1.0):
        raise NotImplementedError("Item rendering must be implemented by subclass")

    def render(self, state, prev_aH, prev_aR, theta, mode='human'):
        print(state)
        gw, gh = 200.0 / self.gridworld.width, 200.0 / self.gridworld.height
        self.gridworld.set_object_positions({'R': state['R']['pos'], 'H': state['H']['pos']})

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
                formula = list('+'.join(recipe) + f'=') + [i]
                for j, c in enumerate(formula):
                    img, tr = self.make_item_image_transform(c, scale=0.5)
                    img.add_attr(header_transform)
                    tr.set_translation(hl*j, 1.2*hl*i)
                    self.gridworld.viewer.add_geom(img)

            ### Recipe objects for thought bubble
            self.recipe_images = []
            for recipe_idx in range(len(self.recipes)):
                recipe, recipe_transform = self.make_item_image_transform(recipe_idx, scale=0.4)
                recipe.add_attr(self.human_transform)
                recipe_transform.set_translation(gw, gh)
                self.recipe_images.append(recipe)

        self.gridworld.render(mode=mode)

        human_coords = self.gridworld.grid.coords_from_pos(state['H']['pos'])
        self.human_transform.set_translation(*human_coords)
        robot_coords = self.gridworld.grid.coords_from_pos(state['R']['pos'])
        self.robot_transform.set_translation(*robot_coords)

        ### Render recipe in thought bubble
        self.gridworld.viewer.add_onetime(self.recipe_images[theta])


        ### Render hand content
        def render_hand(hand, player_transform):
            if hand != self.EMPTY_HAND:
                item, transform = self.make_item_image_transform(hand, scale=0.4)
                transform.set_translation(0, -5)
                item.add_attr(player_transform)
                self.gridworld.viewer.add_onetime(item)

        render_hand(state['H']['hand'], self.human_transform)
        render_hand(state['R']['hand'], self.robot_transform)

        ### Render plate content
        (plate_pos,) = self.gridworld.get_layout_positions('P')
        plate_coords = self.gridworld.grid.coords_from_pos(plate_pos)
        if state['recipe'] is not None:
            for idx, recipe in enumerate(self.recipes):
                if state['recipe'] == recipe:
                    recipe, transform = self.make_item_image_transform(idx, scale=0.65)
                    transform.set_translation(*plate_coords)
                    self.gridworld.viewer.add_onetime(recipe)
                    recipe_made = True
                    break
        else:
            for j, item_name in enumerate(state['plate']):
                item, transform = self.make_item_image_transform(item_name, scale=0.4)

                d = 7
                dx = (-1) ** (j + 1) * d
                dy = (-1) ** (j // 2) * d
                item_coords = (lambda x, y: (x+dx, y+dy))(*plate_coords)

                transform.set_translation(*item_coords)
                self.gridworld.viewer.add_onetime(item)

        return self.gridworld.viewer.render(return_rgb_array=mode == 'rgb_array')


class CakeOrPieGridworld(AbstractRecipeGridworld):
    def __init__(self):
        layout = [
            "XXPXX",
            "C   B",
            "D X X",
            "X X X",
            "X X X",
            "XXXAX",
        ][::-1]  # Reverse so that index 0 means what is visually the bottom row
        player_positions = {'R': (3, 1), 'H': (1, 3)}
        rendering_fns = {
            'H': [make_ellipse_renderer(scale_width=0.8, scale_height=0.56, offset=(1, 1), rgb_color=(0.9, 0.9, 0.9)),
                  make_ellipse_renderer(scale_width=0.17, scale_height=0.12, offset=(0.6, 0.6), rgb_color=(0.9, 0.9, 0.9)),
                  make_ellipse_renderer(scale_width=0.1, scale_height=0.07, offset=(0.4, 0.4), rgb_color=(0.9, 0.9, 0.9)),
                  make_image_renderer('images/girl1.png')],
            'R': [make_image_renderer('images/robot1.png')],
            'X': [make_cell_renderer((0.8, 0.75, 0.7))],
            'A': [make_cell_renderer((0.8, 0.75, 0.7)),
                  make_image_renderer('images/sugar1.png', scale=0.9)],
            'B': [make_cell_renderer((0.8, 0.75, 0.7)),
                  make_image_renderer('images/cherry1.png', scale=0.9, rgb_color=(0.5, 0.7, 0.0))],
            'C': [make_cell_renderer((0.8, 0.75, 0.7)),
                  make_image_renderer('images/chocolate2.png', scale=0.9)],
            'D': [make_cell_renderer((0.8, 0.75, 0.7)),
                  make_image_renderer('images/dough2.png', scale=0.9)],
            'P': [make_cell_renderer((0.8, 0.75, 0.7)),
                  make_image_renderer('images/plate1.png')],
        }
        gridworld = Gridworld(layout, player_positions, rendering_fns)

        initial_state = {
            'H': {
                'pos': (1, 3),
                'or': Direction.NORTH,
                'hand': self.EMPTY_HAND,
            },
            'R': {
                'pos': (3, 1),
                'or': Direction.NORTH,
                'hand': self.EMPTY_HAND,
            },
            'plate': (),
            'recipe': None,
        }

        recipes = [
            ('D', 'B', 'A', 'C'),
            ('D', 'B'),
        ]
        ingredients = ['A', 'B', 'C', 'D']
        # 50% recipe 0 (cake), 50% recipe 1 (pie)
        recipe_distribution = UniformDiscreteDistribution([0, 1])
        super().__init__(gridworld, initial_state, ingredients, recipes, recipe_distribution)

    def get_human_action_distribution(self, obsH, prev_aR, theta):
        obj_pos = {'H': obsH['H']['pos'], 'R': obsH['R']['pos']}
        obj_or = {'H': obsH['H']['or'], 'R': obsH['R']['or']}
        wrap_direction = lambda d: KroneckerDistribution(Direction.get_number_from_direction(d))

        def plan_to_interact(wall_type):
            facing = self.gridworld.get_facing_wall_type('H', obj_pos, obj_or)
            if facing == wall_type:
                return KroneckerDistribution(self.INTERACT)
            # TODO: Have some actual path planning, not this myopic nonsense
            # TODO: Don't assume there's only one location for each ingredient
            # Find the direction to the desired wall type
            hx, hy = obsH['H']['pos']
            ((wallx, wally),) = self.gridworld.get_layout_positions(wall_type)
            composite_direction = (wallx - hx, wally - hy)
            # Decompose into the one or two directions that get us closer
            potential_directions = Direction.get_component_directions(composite_direction)
            # Take whichever one doesn't run into a wall (unless it's needed to
            # change orientation to face the correct wall type)
            for direction in potential_directions:
                next_pos = Direction.move_in_direction((hx, hy), direction)
                if self.gridworld.get_layout_type(next_pos) in [' ', wall_type]:
                    return wrap_direction(direction)
            # If nothing works, just stay put and hope something changes
            return wrap_direction(Direction.STAY)

        # If we've picked up something, put it on the plate
        if obsH['H']['hand'] != self.EMPTY_HAND:
            return plan_to_interact('P')

        # TODO: Pass pedagogic vs not flag as a parameter
        # For cake, human pedagogically selects chocolate first
        ingredient_order = ['C', 'D', 'B', 'A'] if theta == 0 else ['D', 'B']
        # Unpedagogic baseline
        # ingredient_order = ['D', 'B', 'C', 'A'] if theta == 0 else ['D', 'B']
        for ingredient in ingredient_order:
            if ingredient != obsH['R']['hand'] and ingredient not in obsH['plate']:
                return plan_to_interact(ingredient)

        return wrap_direction(Direction.STAY)

    def make_item_image_transform(self, item, scale=1):

        gw, gh = 200.0 / self.gridworld.width, 200.0 / self.gridworld.height
        def make_image_transform(filename, scale=1, w=1.0, h=None, c=None):
            if h is None: h = w
            fullname = get_asset(f'images/{filename}')
            img = rendering.Image(fullname, scale * w * gw, scale * h * gh)
            transform = rendering.Transform()
            img.add_attr(transform)

            if c is not None:
                img.set_color(*c)

            return img, transform

        mapping = {
            'A' : partial(make_image_transform, 'sugar1.png', c=(0.9, 0.9, 0.9)),
            'B' : partial(make_image_transform, 'cherry1.png'),
            'C' : partial(make_image_transform, 'chocolate2.png'),
            'D' : partial(make_image_transform, 'dough2.png'),
            'P' : partial(make_image_transform, 'plate1.png', w=1.3, h=1.3),
            '+' : partial(make_image_transform, 'plus1.png', w=0.5, h=0.5),
            '=' : partial(make_image_transform, 'equal1.png', w=0.5, h=0.2),
            0 : partial(make_image_transform, 'cake2.png', c=(0.8, 0.4, 0.1)),
            1 : partial(make_image_transform, 'apple-pie1.png', c=(0.7, 0.3, 0.2)),
        }
        return mapping[item](scale=scale)

def get_cake_or_pie_hardcoded_robot_policy(*args, **kwargs):
    class Policy:
        def predict(self, ob, state=None):
            N, S, E, W, STAY = Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST, Direction.STAY
            N, S, E, W, STAY = map(Direction.get_number_from_direction, [N, S, E, W, STAY])
            I = 5

            # Hacky way of detecting a reset
            if state is None:
                self.actions = [S, S, 'ob']

            if len(self.actions) == 0:
                return STAY, 'ignored'

            if self.actions[0] == 'ob':
                if ob[1][4][1] == 1.0: # H went towards chocolate
                    self.actions = [I, N, N, N, W, N, I, E, I, W, N, I, I, I, I]
                else:
                    self.actions = [N, N, N, E, I, W, N, I, I, I, I]

            aR = self.actions.pop(0)
            assert aR in range(6)
            return aR, 'ignored'

    return Policy()
