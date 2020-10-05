from copy import deepcopy
from functools import partial
from gym.spaces import Discrete, Box
import numpy as np

import assistance_games.rendering as rendering
from assistance_games.utils import get_asset, MOVEMENT_ACTIONS
from assistance_games.core import AssistancePOMDP, UniformContinuousDistribution, KroneckerDistribution


class SmallPieGridworld(AssistancePOMDP):
    def __init__(self, discrete=False):
        assert not discrete, "Discretized version not yet implemented"
        initial_state = {
            'human_pos' : (0, 2),
            'human_hand' : '',
            'robot_pos' : (0, 0),
            'robot_hand' : '',
            'plate' : (),
            'pie' : None,
            'prev_h_action': 0,
            'prev_r_action': 0,
        }
        self.width = 5
        self.height = 3
        self.recipes = [
            ('A',),
            ('B',),
            ('C',)
        ]
        self.counter_items = {
            (3, 0) : 'A',
            (2, 0) : 'B',
            (1, 0) : 'C',
            (4, 1) : 'P',
        }
        num_ingredients = len(self.counter_items) - 1
        num_dims = self.width + self.height + 2 * num_ingredients + 1
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
        self.viewer = None

    def get_transition_distribution(self, state, human_action, robot_action):
        s = deepcopy(state)

        # s['human_pos'] = self.update_pos(state['human_pos'], human_action)
        s['robot_pos'] = self.update_pos(state['robot_pos'], robot_action)

        # s['human_hand'] = self.update_hand(state['human_pos'], state['human_hand'], human_action)
        s['robot_hand'] = self.update_hand(state['robot_pos'], state['robot_hand'], robot_action)

        # s['plate'] = self.update_plate(state['plate'], state['human_pos'], state['human_hand'], human_action)
        s['plate'] = self.update_plate(s['plate'], state['robot_pos'], state['robot_hand'], robot_action)

        s['pie'] = self.update_pie(s['plate'], state['robot_pos'], state['robot_hand'], robot_action)

        s['prev_h_action'] = human_action
        s['prev_r_action'] = robot_action

        return KroneckerDistribution(s)


    def update_pos(self, pos, act):
        if act >= len(MOVEMENT_ACTIONS):
            return pos

        x, y = pos
        dx, dy = MOVEMENT_ACTIONS[act]
        new_x = np.clip(x + dx, 0, self.width - 1)
        new_y = np.clip(y + dy, 0, self.height - 1)

        return new_x, new_y


    def update_hand(self, pos, hand, action):
        if action == self.INTERACT and pos in self.counter_items:
            if hand == '' and self.counter_items[pos] != 'P':
                return self.counter_items[pos]
            if self.counter_items[pos] in (hand, 'P'):
                return ''
        return hand


    def update_plate(self, plate, pos, hand, action):
        if hand != '' and action == self.INTERACT and self.counter_items.get(pos, '') == 'P' and len(plate) < 4:
            new_plate = list(plate)
            new_plate.append(hand)
            new_plate = tuple(sorted(new_plate))
            return new_plate
        return plate

    def update_pie(self, plate, pos, hand, action):
        if hand == '' and action == self.INTERACT and self.counter_items.get(pos, '') == 'P' and plate in self.recipes:
            return plate
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
        num_ingredients = len(self.counter_items) - 1
        def one_hot(i, n):
            return np.eye(n)[i]

        def position_ob(pos):
            x, y = pos
            return np.concatenate([
                one_hot(x, self.width),
                one_hot(y, self.height),
            ])

        def item_idx(item):
            return ord(item) - ord('A')

        def hand_ob(hand):
            if hand == '':
                return np.zeros(num_ingredients)
            else:
                return one_hot(item_idx(hand), num_ingredients)

        def plate_ob(plate):
            ob = np.zeros(num_ingredients)
            for item in plate:
                ob[item_idx(item)] = 1.0
            return ob

        return KroneckerDistribution(np.concatenate([
            # position_ob(state['human_pos']),
            position_ob(obs['robot_pos']),
            # hand_ob(state['human_hand']),
            hand_ob(obs['robot_hand']),
            plate_ob(obs['plate']),
            np.array([obs['prev_h_action']]),
        ]))

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
        super().close()

    def render(self, state, prev_aH, prev_aR, theta, mode='human'):
        print(state)

        width = self.width
        height = self.height

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

        def move_to_counter(pos):
            x, y = pos
            if x == 0:
                return (-1, y)
            elif y == 0:
                return (x, -1)
            elif x == width - 1:
                return (width, y)
            elif y == height - 1:
                return (x, height)
            else:
                return (x, y)


        if self.viewer is None:
            self.viewer = rendering.Viewer(500,800)
            self.viewer.set_bounds(-130, 120, -150, 250)

            g_x0 = -110 + grid_side
            g_y0 = -110 + grid_side

            self.grid = rendering.Grid(start=(g_x0, g_y0), grid_side=grid_side, shape=(width, height))
            self.grid.set_color(0.85, 0.85, 0.85)
            self.viewer.add_geom(self.grid)

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


            ### Render counters

            make_rect = lambda x, y, w, h : rendering.make_polygon([(x,y),(x+w,y),(x+w,y+h),(x,y+h)])

            make_grid_rect = lambda i, j, di, dj : make_rect(g_x0 + i*gs, g_y0 + j*gs, di*gs, dj*gs)

            counters = [
                make_grid_rect(-1, -1, width+2, 1),
                make_grid_rect(-1, -1, 1, height+2),

                make_grid_rect(-1, height, width+2, 1),
                make_grid_rect(width, -1, 1, height+2),
            ]

            for counter in counters:
                r = 0.8
                off = 0.05
                g = r - off
                b = r - 2 * off
                counter.set_color(r, g, b)
                self.viewer.add_geom(counter)

            for pos, itemname in self.counter_items.items():
                counter_pos = move_to_counter(pos)
                coords = self.grid.coords_from_pos(counter_pos)

                item, transform = make_item_image[itemname]()
                transform.set_translation(*coords)
                self.viewer.add_geom(item)


            ### Render formulae

            g_x0 = -110 + grid_side
            g_y0 = -110 + grid_side
            header_x = g_x0 + 0 * grid_side
            header_y = g_y0 + (height + 2) * grid_side
            hl = 15

            header_transform = rendering.Transform()
            header_transform.set_translation(header_x, header_y)

            for i, recipe in enumerate(self.recipes):
                formula = '+'.join(recipe) + f'={i}'
                for j, c in enumerate(formula):
                    img, tr = make_item_image[c](s=0.5)
                    img.add_attr(header_transform)
                    tr.set_translation(hl*j, 1.2*hl*i)
                    self.viewer.add_geom(img)


            # Render thought bubble

            scale = 0.8
            thought = rendering.make_ellipse(scale * grid_side/2, scale * 0.7*grid_side/2)
            thought.set_color(0.9, 0.9, 0.9)
            thought_transform = rendering.Transform()
            thought_transform.set_translation(-gs, gs)
            thought.add_attr(thought_transform)
            thought.add_attr(self.human_transform)

            self.viewer.add_geom(thought)

            scale = 0.17
            thought2 = rendering.make_ellipse(scale * grid_side/2, scale * grid_side/2)
            thought2.set_color(0.9, 0.9, 0.9)
            thought_transform2 = rendering.Transform()
            thought_transform2.set_translation(-0.6 * gs, 0.6 * gs)
            thought2.add_attr(thought_transform2)
            thought2.add_attr(self.human_transform)

            self.viewer.add_geom(thought2)

            scale = 0.1
            thought3 = rendering.make_ellipse(scale * grid_side/2, scale * grid_side/2)
            thought3.set_color(0.9, 0.9, 0.9)
            thought_transform3 = rendering.Transform()
            thought_transform3.set_translation(-0.4 * gs, 0.4 * gs)
            thought3.add_attr(thought_transform3)
            thought3.add_attr(self.human_transform)

            self.viewer.add_geom(thought3)

            self.pies = []
            for pie_idx in ('012'):
                pie, _ = make_item_image[pie_idx](s=0.4)
                pie.add_attr(self.human_transform)
                pie.add_attr(thought_transform)
                self.pies.append(pie)


        human_pos = state['human_pos']
        robot_pos = state['robot_pos']
        # human_hand = state['human_hand']
        robot_hand = state['robot_hand']
        plate = state['plate']
        pie = state['pie']
        preferred_idx = max(range(len(theta)), key=lambda i: theta[i]) # argmax

        human_coords = self.grid.coords_from_pos(human_pos)
        self.human_transform.set_translation(*human_coords)

        robot_coords = self.grid.coords_from_pos(robot_pos)
        self.robot_transform.set_translation(*robot_coords)

        
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

        items_to_pos = {item:pos for pos, item in self.counter_items.items()}
        plate_pos = move_to_counter(items_to_pos['P'])
        plate_coords = self.grid.coords_from_pos(plate_pos)


        ### Render plate content

        if pie is not None:
            for idx, recipe in enumerate(self.recipes):
                if pie == recipe:
                    pie, transform = make_item_image[str(idx)](s=0.65)
                    transform.set_translation(*plate_coords)
                    self.viewer.add_onetime(pie)
                    recipe_made = True
                    break
        else:
            for j, item_name in enumerate(plate):
                item, transform = make_item_image[item_name](s=0.4)

                d = 7
                dx = (-1) ** (j+1) * d
                dy = (-1) ** (j // 2) * d
                item_coords = (lambda x, y : (x+dx, y+dy))(*plate_coords)

                transform.set_translation(*item_coords)
                self.viewer.add_onetime(item)


        ### Render pie in thought bubble
        self.viewer.add_onetime(self.pies[preferred_idx])


        return self.viewer.render(return_rgb_array = mode=='rgb_array')
