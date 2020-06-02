from collections import Counter
from copy import deepcopy
from functools import partial

from gym.spaces import Discrete, Box
import numpy as np

from assistance_games.core import (
    AssistanceGame,
    AssistanceProblem,
    get_human_policy,
    functional_random_policy_fn,
    FunctionalObservationModel,
    FunctionalTransitionModel,
    FunctionalRewardModel,
    SensorModel,
)

import assistance_games.rendering as rendering
from assistance_games.utils import get_asset


class PieGridworldAssistanceGame(AssistanceGame):
    def __init__(self):
        self.width = 5
        self.height = 6

        self.counter_items = {
            (0, 2) : 'A',
            (4, 5) : 'B',
            (0, 4) : 'C',
            (0, 5) : 'D',
            (4, 1) : 'E',
            (4, 2) : 'F',
            (2, 5) : 'P',
        }


        self.recipes = [
            ('A', 'A', 'B', 'B'),
            ('A', 'B', 'C', 'E'),
            ('A', 'B', 'D', 'F'),
        ]

        human_action_space = Discrete(6)
        robot_action_space = Discrete(6)

        self.INTERACT = 5

        state_space = None
        self.initial_state = {
            'human_pos' : (0, 2),
            'human_hand' : '',
            'robot_pos' : (4, 2),
            'robot_hand' : '',
            'plate' : (),
        }

        horizon = 20
        discount = 0.95

        rewards_dist = []
        num_recipes = len(self.recipes)
        for idx in range(num_recipes):
            reward_fn = partial(self.reward_fn, reward_idx=idx)
            rewards_dist.append((reward_fn, 1/num_recipes))
        

        super().__init__(
            state_space=state_space,
            human_action_space=human_action_space,
            robot_action_space=robot_action_space,
            transition=self.transition_fn,
            reward_distribution=rewards_dist,
            horizon=horizon,
            discount=discount,
        )

    def reward_fn(self, state, next_state=None, human_action=0, robot_action=0, reward_idx=0):
        recipe_reward = int(state['plate'] == self.recipes[reward_idx])
        return recipe_reward

    def transition_fn(self, state, human_action=0, robot_action=0):
        s = deepcopy(state)

        s['human_pos'] = self.update_pos(state['human_pos'], human_action)
        s['robot_pos'] = self.update_pos(state['robot_pos'], robot_action)

        s['human_hand'] = self.update_hand(state['human_pos'], state['human_hand'], human_action)
        s['robot_hand'] = self.update_hand(state['robot_pos'], state['robot_hand'], robot_action)

        s['plate'] = self.update_plate(state['plate'], state['human_pos'], state['human_hand'], human_action)
        s['plate'] = self.update_plate(s['plate'], state['robot_pos'], state['robot_hand'], robot_action)

        return s


    def update_pos(self, pos, act):
        x, y = pos
        dirs = [
            (0, 0),
            (1, 0),
            (0, 1),
            (-1, 0),
            (0, -1),
            (0, 0),
        ]
        dx, dy = dirs[act]

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


class PieGridworldAssistanceProblem(AssistanceProblem):
    def __init__(self, human_policy_fn=functional_random_policy_fn, **kwargs):
        assistance_game = PieGridworldAssistanceGame()

        human_policy_fn = pie_human_policy_fn

        self.ag = assistance_game

        super().__init__(
            assistance_game=assistance_game,
            human_policy_fn=human_policy_fn,

            state_space_builder=pie_state_space_builder,
            transition_model_fn_builder=pie_transition_model_fn_builder,
            reward_model_fn_builder=pie_reward_model_fn_builder,
            sensor_model_fn_builder=pie_sensor_model_fn_builder,
            observation_model_fn=pie_observation_model_fn_builder(assistance_game),
        )

    def render(self, mode='human'):
        print(self.state)

        width = self.ag.width
        height = self.ag.height

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
            'A' : partial(make_image_transform, 'flour4.png'),
            'B' : partial(make_image_transform, 'sugar6.png', w=1.2, h=1.2),
            'C' : partial(make_image_transform, 'chocolate4.png'),
            'D' : partial(make_image_transform, 'chocolate4.png', c=(0.1, 0.1, 0.1)),
            'E' : partial(make_image_transform, 'cherry1.png'),
            'F' : partial(make_image_transform, 'apple3.png', c=(0.7, 0.3, 0.2)),
            'P' : partial(make_image_transform, 'plate1.png', w=1.3, h=1.3),
            '+' : partial(make_image_transform, 'plus1.png', w=0.5, h=0.5),
            '=' : partial(make_image_transform, 'equal1.png', w=0.5, h=0.2),
            '0' : partial(make_image_transform, 'pie1.png'),
            '1' : partial(make_image_transform, 'cake1.png'),
            '2' : partial(make_image_transform, 'apple-pie1.png', c=(0.7, 0.3, 0.2)),
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

        g_x0 = -110 + grid_side
        g_y0 = -110 + grid_side

        make_rect = lambda x, y, w, h : rendering.make_polygon([(x,y),(x+w,y),(x+w,y+h),(x,y+h)])

        make_grid_rect = lambda i, j, di, dj : make_rect(g_x0 + i*gs, g_y0 + j*gs, di*gs, dj*gs)


        if self.viewer is None:
            self.viewer = rendering.Viewer(500,800)
            self.viewer.set_bounds(-130, 120, -150, 250)

            grid_background = make_grid_rect(0, 0, width, height)
            grid_background.set_color(0.92, 0.92, 0.92)
            self.viewer.add_geom(grid_background)

            self.grid = rendering.Grid(start=(g_x0, g_y0), grid_side=grid_side, shape=(width, height))
            self.grid.set_color(0.85, 0.85, 0.85)
            self.viewer.add_geom(self.grid)

            human_image = get_asset('images/girl9-red2.png')
            human = rendering.Image(human_image, grid_side, grid_side)
            self.human_transform = rendering.Transform()
            human.add_attr(self.human_transform)
            self.viewer.add_geom(human)

            robot_image = get_asset('images/robot10.png')
            robot = rendering.Image(robot_image, grid_side, grid_side)
            self.robot_transform = rendering.Transform()
            robot.add_attr(self.robot_transform)
            self.viewer.add_geom(robot)


            ### Render counters

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

            for pos, itemname in self.ag.counter_items.items():
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
            hl = 18

            header_transform = rendering.Transform()
            header_transform.set_translation(header_x, header_y)

            for i, recipe in enumerate(self.ag.recipes):
                formula = '+'.join(recipe) + f'={i}'
                for j, c in enumerate(formula):
                    img, tr = make_item_image[c](s=0.6)
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


        human_pos = self.state['human_pos']
        robot_pos = self.state['robot_pos']
        human_hand = self.state['human_hand']
        robot_hand = self.state['robot_hand']
        plate = self.state['plate']
        reward_idx = self.state['reward_idx']

        human_coords = self.grid.coords_from_pos(human_pos)
        self.human_transform.set_translation(*human_coords)

        robot_coords = self.grid.coords_from_pos(robot_pos)
        self.robot_transform.set_translation(*robot_coords)

        
        ### Render hand content

        for hand, hand_transform in (
            (human_hand, self.human_transform),
            (robot_hand, self.robot_transform),
        ):
            if hand != '':
                item, transform = make_item_image[hand](s=0.4)
                transform.set_translation(0, -5)
                item.add_attr(hand_transform)
                self.viewer.add_onetime(item)

        items_to_pos = {item:pos for pos, item in self.ag.counter_items.items()}
        plate_pos = move_to_counter(items_to_pos['P'])
        plate_coords = self.grid.coords_from_pos(plate_pos)


        ### Render plate content

        recipe_made = False
        for idx, recipe in enumerate(self.ag.recipes):
            if plate == recipe:
                pie, transform = make_item_image[str(idx)](s=0.65)
                transform.set_translation(*plate_coords)
                self.viewer.add_onetime(pie)
                recipe_made = True
                break


        if not recipe_made:
            for j, item_name in enumerate(plate):
                item, transform = make_item_image[item_name](s=0.4)

                d = 7
                dx = (-1) ** (j+1) * d
                dy = (-1) ** (j // 2) * d
                item_coords = (lambda x, y : (x+dx, y+dy))(*plate_coords)

                transform.set_translation(*item_coords)
                self.viewer.add_onetime(item)


        ### Render pie in thought bubble
        self.viewer.add_onetime(self.pies[reward_idx])


        return self.viewer.render(return_rgb_array = mode=='rgb_array')


class PieGridworldProblemStateSpace:
    def __init__(self, ag):
        self.initial_state = ag.initial_state
        self.num_rewards = len(ag.reward_distribution)

    def sample_initial_state(self):
        state = deepcopy(self.initial_state)
        state['reward_idx'] = np.random.randint(self.num_rewards)
        return state

def pie_state_space_builder(ag):
    return PieGridworldProblemStateSpace(ag)

def pie_transition_model_fn_builder(ag, human_policy_fn):
    def transition_fn(state, action):
        human_policy = human_policy_fn(ag, state['reward_idx'])
        return ag.transition(state, human_policy(state), action)

    transition_model_fn = partial(FunctionalTransitionModel, fn=transition_fn)
    return transition_model_fn

def pie_reward_model_fn_builder(ag, human_policy_fn):
    def reward_fn(state, action=None, next_state=None):
        reward = ag.reward_distribution[state['reward_idx']][0]
        return reward(state=state, next_state=next_state)

    reward_model_fn = partial(FunctionalRewardModel, fn=reward_fn)
    return reward_model_fn

def pie_sensor_model_fn_builder(ag, human_policy_fn):
    return SensorModel

def pie_observation_model_fn_builder(ag):
    num_ingredients = len(ag.counter_items) - 1

    def observation_fn(state, action=None, sense=None):
        def one_hot(i, n):
            return np.eye(n)[i]

        def position_ob(pos):
            x, y = pos
            return np.concatenate([
                one_hot(x, ag.width),
                one_hot(y, ag.height),
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

        return np.concatenate([
            position_ob(state['human_pos']),
            position_ob(state['robot_pos']),
            hand_ob(state['human_hand']),
            hand_ob(state['robot_hand']),
            plate_ob(state['plate']),
        ])

    pos_ob_dims = ag.width + ag.height
    num_dims = 2 * pos_ob_dims + 3 * num_ingredients
    low = np.zeros(num_dims)
    high = np.ones(num_dims)
    high[:4] = [ag.width, ag.height, ag.width, ag.height]

    ob_space = Box(low=low, high=high)

    observation_model_fn = partial(FunctionalObservationModel, fn=observation_fn, space=ob_space)
    return observation_model_fn


def get_pie_hardcoded_robot_policy(*args, **kwargs):
    class Policy:
        def predict(self, ob, state=None):
            t, r_idx = state if state is not None else (0, None)

            width = 5
            height = 6

            onehot_x = ob[:width]
            onehot_y = ob[width:width+height]

            x = np.argmax(onehot_x)
            y = np.argmax(onehot_y)
            human_pos = (x, y)


            if t == 1:
                if human_pos == (0, 2):
                    r_idx = 0
                else:
                    r_idx = 3

            if r_idx == 3:
                if t == 3 and human_pos == (0, 4):
                    r_idx = 1

                if t == 3 and human_pos == (0, 5):
                    r_idx = 2



            S, R, U, L, D, A = range(6)


            robot_policies = [
                [
                    S,          # Wait 1 step
                    U, U, U, A, # Get dark flour
                    L, L, A,    # Take to plate
                    R, R, A,    # Get dark flour
                    L, L, A,    # Take to plate
                ],

                [
                    S, S, S,             # Wait 1 step
                    D, A,                # get milk chocolate
                    U, U, U, U, L, L, A, # drop in plate
                    R, R, A,             # get dark flour
                    L, L, A,             # Take to plate
                ],

                [
                    S, S, S,                # Wait 1 step
                    A,                      # get milk chocolate
                    U, U, U, L, L, A, # drop in plate
                    R, R, A,                # get dark flour
                    L, L, A,                # Take to plate
                ],
            ]

            if r_idx is None:
                robot_policy = robot_policies[0]
            elif r_idx == 3:
                robot_policy = robot_policies[1]
            else:
                robot_policy = robot_policies[r_idx]


            act = robot_policy[t] if t < len(robot_policy) else S

            return act, (t+1, r_idx)

    return Policy()


def pie_human_policy_fn(ag, reward):
    def human_policy(state):
        S, R, U, L, D, A = range(6)

        # policy 0

        policy0 = [
            A,                # get white flour
            U, U, U, R, R, A, # take to plate
            L, L, D, D, D, A, # get white flour
            U, U, U, R, R, A, # take to plate
            S,
        ]

        trail0 = [
            ((0, 2), '', {'A' : 0}),

            ((0, 2), 'A', {'A' : 0}),
            ((0, 3), 'A', {'A' : 0}),
            ((0, 4), 'A', {'A' : 0}),
            ((0, 5), 'A', {'A' : 0}),
            ((1, 5), 'A', {'A' : 0}),
            ((2, 5), 'A', {'A' : 0}),

            ((2, 5), '', {'A' : 1}),
            ((1, 5), '', {'A' : 1}),
            ((0, 5), '', {'A' : 1}),
            ((0, 4), '', {'A' : 1}),
            ((0, 3), '', {'A' : 1}),
            ((0, 2), '', {'A' : 1}),

            ((0, 2), 'A', {'A' : 1}),
            ((0, 3), 'A', {'A' : 1}),
            ((0, 4), 'A', {'A' : 1}),
            ((0, 5), 'A', {'A' : 1}),
            ((1, 5), 'A', {'A' : 1}),
            ((2, 5), 'A', {'A' : 1}),

            ((2, 5), '', {'A' : 2}),
        ]


        policy1 = [
            U, U, A, # get green apple
            U, R, R, A, # drop in plate
            L, L, D, D, D, A, # get white flour
            U, U, U, R, R, A, # take to plate
            S,
        ]

        trail1 = [
            ((0, 2), '',  {'C' : 0}),
            ((0, 3), '',  {'C' : 0}),
            ((0, 4), '',  {'C' : 0}),
            ((0, 4), 'C', {'C' : 0}),
            ((0, 5), 'C', {'C' : 0}),
            ((1, 5), 'C', {'C' : 0}),
            ((2, 5), 'C', {'C' : 0}),

            ((2, 5), '', {'A' : 0, 'C' : 1}),
            ((1, 5), '', {'C' : 1}),
            ((0, 5), '', {'C' : 1}),
            ((0, 4), '', {'C' : 1}),
            ((0, 3), '', {'C' : 1}),
            ((0, 2), '', {'C' : 1}),

            ((0, 2), 'A', {'C' : 1}),
            ((0, 3), 'A', {'C' : 1}),
            ((0, 4), 'A', {'C' : 1}),
            ((0, 5), 'A', {'C' : 1}),
            ((1, 5), 'A', {'C' : 1}),
            ((2, 5), 'A', {'C' : 1}),

            ((2, 5), '', {'A' : 1, 'C' : 1}),
        ]



        policy2 = [
            U, U, U, A, # get red apple
            R, R, A, # drop in plate
            L, L, D, D, D, A, # get white flour
            U, U, U, R, R, A, # take to plate
            S,
        ]

        trail2 = [
            ((0, 2), '',  {'A' : 0, 'D' : 0}),
            ((0, 3), '',  {'A' : 0, 'D' : 0}),
            ((0, 4), '',  {'A' : 0, 'D' : 0}),
            ((0, 5), '',  {'A' : 0, 'D' : 0}),
            ((0, 5), 'D', {'A' : 0, 'D' : 0}),
            ((1, 5), 'D', {'A' : 0, 'D' : 0}),
            ((2, 5), 'D', {'A' : 0, 'D' : 0}),

            ((2, 5), '', {'A' : 0, 'D' : 1}),
            ((1, 5), '', {'D' : 1}),
            ((0, 5), '', {'D' : 1}),
            ((0, 4), '', {'D' : 1}),
            ((0, 3), '', {'D' : 1}),
            ((0, 2), '', {'D' : 1}),

            ((0, 2), 'A', {'D' : 1}),
            ((0, 3), 'A', {'D' : 1}),
            ((0, 4), 'A', {'D' : 1}),
            ((0, 5), 'A', {'D' : 1}),
            ((1, 5), 'A', {'D' : 1}),
            ((2, 5), 'A', {'D' : 1}),

            ((2, 5), '', {'A' : 1, 'D' : 1}),
        ]


        reward_idx = state['reward_idx']

        policy = (policy0, policy1, policy2)[reward_idx]
        trail = (trail0, trail1, trail2)[reward_idx]


        def has_plate_condition(plate, cond):
            c = Counter(plate)
            return all(c[k] == v for k, v in cond.items())


        def get_time(state):
            for t, (pos, hand, plate_cond) in enumerate(trail):
                if (
                    pos == state['human_pos'] and
                    hand == state['human_hand'] and
                    has_plate_condition(state['plate'], plate_cond)
                ):
                    return t

        t = get_time(state)
        action = policy[t] if t is not None else A

        return action

    return human_policy
