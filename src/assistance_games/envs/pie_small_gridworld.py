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
    ShapedFunctionalRewardModel,
    SensorModel,
    TerminationModel,
)

from assistance_games.utils import get_asset


class SmallPieGridworldAssistanceGame(AssistanceGame):
    def __init__(self):
        self.width = 5
        self.height = 3

        self.counter_items = {
            (3, 0) : 'A',
            (2, 0) : 'B',
            (1, 0) : 'C',
            (4, 1) : 'P',
        }


        self.recipes = [
            ('A',),
            ('B',),
            ('C',)
        ]

        num_questions = len(self.recipes)
        human_action_space = Box(0, 3, shape=())
        robot_action_space = Discrete(6 + num_questions)

        self.INTERACT = 5

        state_space = None
        self.initial_state = {
            'human_pos' : (0, 2),
            'human_hand' : '',
            'robot_pos' : (0, 0),
            'robot_hand' : '',
            'plate' : (),
            'pie' : None,
            'prev_h_action': 0,
            'prev_r_action': 0,
        }

        horizon = 20
        discount = 0.99

        rewards_dist = {
            ('A',) : (0, 0), # TODO: Eventually we want (1, 3) here
            ('B',) : (0, 2),
            ('C',) : (0, 2)
        }

        super().__init__(
            state_space=state_space,
            human_action_space=human_action_space,
            robot_action_space=robot_action_space,
            transition=self.transition_fn,
            reward_distribution=rewards_dist,
            horizon=horizon,
            discount=discount,
        )

    def transition_fn(self, state, human_action=0, robot_action=0):
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

        return s


    def update_pos(self, pos, act):
        dirs = [
            (0, 0),
            (1, 0),
            (0, 1),
            (-1, 0),
            (0, -1),
        ]
        if act >= len(dirs):
            return pos

        x, y = pos
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

    def update_pie(self, plate, pos, hand, action):
        if hand == '' and action == self.INTERACT and self.counter_items.get(pos, '') == 'P' and plate in self.recipes:
            return plate
        return None


class SmallPieGridworldAssistanceProblem(AssistanceProblem):
    def __init__(self, human_policy_fn=functional_random_policy_fn, **kwargs):
        assistance_game = SmallPieGridworldAssistanceGame()

        human_policy_fn = small_pie_human_policy_fn

        self.ag = assistance_game

        super().__init__(
            assistance_game=assistance_game,
            human_policy_fn=human_policy_fn,

            state_space_builder=small_pie_state_space_builder,
            transition_model_fn_builder=small_pie_transition_model_fn_builder,
            reward_model_fn_builder=small_pie_reward_model_fn_builder,
            sensor_model_fn_builder=small_pie_sensor_model_fn_builder,
            observation_model_fn=small_pie_observation_model_fn_builder(assistance_game),
            termination_model_fn_builder=small_pie_termination_model_fn_builder,
        )

    def render(self, mode='human'):
        import assistance_games.rendering as rendering

        print(self.state)

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
            hl = 15

            header_transform = rendering.Transform()
            header_transform.set_translation(header_x, header_y)

            for i, recipe in enumerate(self.ag.recipes):
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


        human_pos = self.state['human_pos']
        robot_pos = self.state['robot_pos']
        # human_hand = self.state['human_hand']
        robot_hand = self.state['robot_hand']
        plate = self.state['plate']
        pie = self.state['pie']
        preferred_idx = self.state['preferred_idx']

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

        items_to_pos = {item:pos for pos, item in self.ag.counter_items.items()}
        plate_pos = move_to_counter(items_to_pos['P'])
        plate_coords = self.grid.coords_from_pos(plate_pos)


        ### Render plate content

        if pie is not None:
            for idx, recipe in enumerate(self.ag.recipes):
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


class SmallPieGridworldProblemStateSpace:
    def __init__(self, ag):
        self.initial_state = ag.initial_state
        self.thetas_dists = [ag.reward_distribution[recipe] for recipe in ag.recipes]

    def sample_initial_state(self):
        state = deepcopy(self.initial_state)
        thetas = [np.random.uniform(low, high) for low, high in self.thetas_dists]
        state['thetas'] = thetas
        state['preferred_idx'] = max(range(len(thetas)), key=lambda i: thetas[i]) # argmax
        return state

def small_pie_state_space_builder(ag):
    return SmallPieGridworldProblemStateSpace(ag)

def small_pie_transition_model_fn_builder(ag, human_policy_fn):
    def transition_fn(state, action):
        human_policy = human_policy_fn(ag, state['thetas'])
        return ag.transition(state, human_policy(state), action)

    transition_model_fn = partial(FunctionalTransitionModel, fn=transition_fn)
    return transition_model_fn

def small_pie_reward_model_fn_builder(ag, human_policy_fn):
    recipes = ag.recipes
    
    def reward_fn(state, action=None, next_state=None):
        if next_state['pie'] is None:
            return 0

        idx = [i for i, recipe in enumerate(recipes) if recipe == next_state['pie']][0]
        return 10 if idx == next_state['preferred_idx'] else -1

    def is_subset(x, y):
        # TODO: Handle duplicates
        return all(a in y for a in x)
    
    def shaping_fn(state):
        total = 0.0
        hand = state['robot_hand']
        plate = state['plate']
        preferred_idx = state['preferred_idx']
        # TODO: Should check whether the item in hand is actually useful (e.g. not already on the plate, contributes to some recipe)
        if hand == '':
            pass
        elif hand in recipes[preferred_idx]:
            total += 1.0
        else:
            total += 0.1

        if plate == '':
            pass
        elif is_subset(plate, recipes[preferred_idx]):
            total += 2.0 * len(plate)
        else:
            # TODO: Should check that the plate corresponds to some possible recipe
            total += 0.2 * len(plate)

        return total

    # To disable reward shaping, simply pass in shaping_fns=[]
    reward_model_fn = partial(ShapedFunctionalRewardModel, fn=reward_fn, shaping_fns=[shaping_fn])
    return reward_model_fn

def small_pie_sensor_model_fn_builder(ag, human_policy_fn):
    return SensorModel

def small_pie_observation_model_fn_builder(ag):
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
            # position_ob(state['human_pos']),
            position_ob(state['robot_pos']),
            # hand_ob(state['human_hand']),
            hand_ob(state['robot_hand']),
            plate_ob(state['plate']),
            np.array([state['prev_h_action']]),
        ])

    num_dims = ag.width + ag.height + 2 * num_ingredients + 1
    low = np.zeros(num_dims)
    high = np.ones(num_dims)
    high[-1] = max(map(max, ag.reward_distribution.values()))

    ob_space = Box(low=low, high=high)

    observation_model_fn = partial(FunctionalObservationModel, fn=observation_fn, space=ob_space)
    return observation_model_fn


class SmallPieTerminationModel(TerminationModel):
    def __call__(self):
        return self.pomdp.state['pie'] != None

def small_pie_termination_model_fn_builder(ag, human_policy):
    return SmallPieTerminationModel


def small_pie_human_policy_fn(ag, thetas):
    def human_policy(state):
        question = state['prev_r_action'] - 6
        return thetas[question] if question >= 0 else 0
    return human_policy


def get_small_pie_hardcoded_robot_policy(*args, **kwargs):
    class Policy:
        def __init__(self):
            S, R, U, L, D, A, QA, QB, QC = range(9)
            self.actions = [R, A, R, R, R, U, A, A]

        def predict(self, ob, state=None):
            if not self.actions:
                return 0, None
            return self.actions.pop(0), None

    return Policy()
