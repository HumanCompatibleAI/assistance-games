from collections import Counter
from copy import deepcopy
import functools
from functools import partial
import itertools

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
    hard_value_iteration,
    soft_value_iteration,
    softhard_value_iteration,
)

import assistance_games.rendering as rendering
from assistance_games.utils import get_asset, dict_to_sparse, sample_distribution


class MiniPieGridworldAssistanceGame(AssistanceGame):
    def __init__(self):
        self.width = 3
        self.height = 4

        self.num_pos = 9

        self.counter_items = {
            2 : 'A',
            3 : 'C',
            5 : 'B',
            8 : 'D',
            4 : 'P',
        }
        self.item_to_counter = {name : pos for pos, name in self.counter_items.items()}


        self.recipes = [
            ('A', 'B'),
            ('A', 'B', 'C', 'D'),
            # ('A', 'B', 'D', 'F'),
        ]

        add_null_reward = True
        add_null_reward = False

        if add_null_reward:
            self.recipes.append(('X',))


        human_action_space = Discrete(3)
        robot_action_space = Discrete(3)

        self.INTERACT = 2

        state_space = None
        self.initial_state = {
            'human_pos' : 1,
            'human_hand' : '',
            'robot_pos' : 8,
            'robot_hand' : '',
            'plate' : (),
            'recipe_made' : False,
        }

        horizon = 20
        discount = 0.95


        self.poss = [
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),

            (1, 3),

            (2, 3),
            (2, 2),
            (2, 1),
            (2, 0),
        ]


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

    def reward_fn1(self, state, next_state=None, human_action=0, robot_action=0, reward_idx=0):
        if next_state is None:
            next_state = state


        def shaping_fn(state):
            def is_subset(plate, recipe):
                plate_counts = Counter(plate)
                recipe_counts = Counter(plate)
                return all(plate_counts[k] <= recipe_counts[k] for k in plate_counts)

            def l1_dist(p, q):
                return abs(p - q)
                # return sum(abs(x - z) for x, z in zip(p, q))

            total = 0.0
            hand = state['robot_hand']
            plate = state['plate']
            reward_idx = state['reward_idx']
            # TODO: Should check whether the item in hand is actually useful (e.g. not already on the plate, contributes to some recipe)
            if hand == '':
                pass
            else:
                new_plate = list(plate)
                new_plate.append(hand)
                new_plate = tuple(sorted(new_plate))
                if is_subset(new_plate, self.recipes[reward_idx]):
                    plate_pos = self.item_to_counter['P']
                    dist_to_plate = l1_dist(state['robot_pos'], plate_pos)
                    k = (self.width + self.height) / 2
                    total += 2.0 * np.exp((-1) * dist_to_plate / k)
                else:
                    # total += 0.1
                    total -= 1.0

            if plate == '':
                pass
            elif is_subset(plate, self.recipes[reward_idx]):
                total += 3.0 * len(plate)
            else:
                # TODO: Should check that the plate corresponds to some possible recipe
                total -= 1.0
                # total += 0.2 * len(plate)

            return total


        recipe_reward = 30 * int(state['plate'] == self.recipes[reward_idx] and state['recipe_made']) + shaping_fn(state)
        # recipe_reward = 100 * int(state['plate'] != self.recipes[reward_idx] and next_state['plate'] == self.recipes[reward_idx])
        return recipe_reward

    def reward_fn2(self, state, next_state=None, human_action=0, robot_action=0, reward_idx=0):
        def bonus_reward(state):
            def is_subset(plate, recipe):
                plate_counts = Counter(plate)
                recipe_counts = Counter(plate)
                return all(plate_counts[k] <= recipe_counts[k] for k in plate_counts)

            def l1_dist(p, q):
                return abs(p - q)
                # return sum(abs(x - z) for x, z in zip(p, q))

            total = 0.0
            hand = state['robot_hand']
            plate = state['plate']
            reward_idx = state['reward_idx']
            # TODO: Should check whether the item in hand is actually useful (e.g. not already on the plate, contributes to some recipe)
            if hand == '':
                pass
            else:
                new_plate = list(plate)
                new_plate.append(hand)
                new_plate = tuple(sorted(new_plate))
                if is_subset(new_plate, self.recipes[reward_idx]):
                    plate_pos = self.item_to_counter['P']
                    dist_to_plate = l1_dist(state['robot_pos'], plate_pos)
                    k = (self.width + self.height) / 2
                    total += 2.0 * np.exp((-1) * dist_to_plate / k)
                else:
                    # total += 0.1
                    total -= 1.0

            if plate == '':
                pass
            elif is_subset(plate, self.recipes[reward_idx]):
                total += 2.0 * len(plate)
            else:
                # TODO: Should check that the plate corresponds to some possible recipe
                total -= 10.0
                # total += 0.2 * len(plate)

            return total

        correct_recipe_reward = 10 * int(state['plate'] == self.recipes[reward_idx] and state['recipe_made'])

        reward = correct_recipe_reward + bonus_reward(state)
        return reward

    reward_fn = reward_fn2

    def transition_fn(self, state, human_action=0, robot_action=0):
        s = deepcopy(state)

        human_pos = state['human_pos']
        human_hand = state['human_hand']
        robot_pos = state['robot_pos']
        robot_hand = state['robot_hand']
        plate = state['plate']
        recipe_made = state['recipe_made']

        s['human_pos'] = self.update_pos(human_pos, human_action)
        s['robot_pos'] = self.update_pos(robot_pos, robot_action)

        s['human_hand'] = self.update_hand(human_pos, human_hand, human_action)
        s['robot_hand'] = self.update_hand(robot_pos, robot_hand, robot_action)

        s['plate'] = self.update_plate(plate, recipe_made, human_pos, human_hand, human_action)
        s['plate'] = self.update_plate(s['plate'], recipe_made, robot_pos, robot_hand, robot_action)

        s['recipe_made'] = self.update_recipe(recipe_made, state['plate'], s['plate'], human_pos, human_hand, human_action)
        s['recipe_made'] = self.update_recipe(s['recipe_made'], state['plate'], s['plate'], robot_pos, robot_hand, robot_action)

        return s


    def update_pos(self, pos, act):
        dirs = [
            1,
            -1,
            0,
        ]
        d = dirs[act]

        new_pos = np.clip(pos + d, 0, self.num_pos - 1)
        return new_pos


    def update_hand(self, pos, hand, action):
        if action == self.INTERACT and pos in self.counter_items:
            if hand == '' and self.counter_items[pos] != 'P':
                return self.counter_items[pos]
            if self.counter_items[pos] in (hand, 'P'):
                return ''
        return hand


    def update_plate(self, plate, recipe_made, pos, hand, action):
        if not recipe_made and hand != '' and action == self.INTERACT and self.counter_items.get(pos, '') == 'P' and len(plate) < 4:
            new_plate = list(plate)
            new_plate.append(hand)
            new_plate = tuple(sorted(new_plate))
            return new_plate
        return plate


    def update_recipe(self, recipe_made, plate, new_plate, pos, hand, action):
        return (
            recipe_made or (
                plate in self.recipes and
                plate == new_plate and
                self.counter_items.get(pos, '') == 'P' and
                hand == '' and
                action == self.INTERACT
            )
        )

    def state_to_id(self, state, humanonly=False):
        items = list(self.counter_items.values())[:-1]
        num_items = len(items)

        pos_range = self.num_pos
        hand_range = num_items + 1
        human_ranges = [pos_range, hand_range]

        # Not currently supporting repeated items
        plate_ranges = [2] * num_items

        list_index = lambda l, v : l.index(v) if v in l else -1

        human_pos = state['human_pos']
        human_hand = state['human_hand']
        human_hand_id = 1 + list_index(items, human_hand)
        human_ids = [human_pos, human_hand_id]

        if not humanonly:
            robot_pos = state['robot_pos']
            robot_hand = state['robot_hand']
            robot_hand_id = 1 + list_index(items, robot_hand)
            robot_ids = [robot_pos, robot_hand_id]
            robot_ranges = human_ranges.copy()
        else:
            robot_ids = []
            robot_ranges = []

        plate_ids = [int(item in state['plate']) for item in items]

        ids = [*human_ids, *robot_ids, *plate_ids]
        ranges = [*human_ranges, *robot_ranges, *plate_ranges]

        k = 1
        full_id = 0
        for idx, r in zip(ids, ranges):
            full_id += k * idx
            k *= r

        return full_id

    
    def id_to_state(self, full_id, humanonly=False):
        items = list(self.counter_items.values())[:-1]
        num_items = len(items)

        pos_range = self.num_pos
        hand_range = num_items + 1
        human_ranges = [pos_range, hand_range]

        plate_ranges = [2] * num_items

        if not humanonly:
            robot_ranges = human_ranges.copy()
        else:
            robot_ranges = []

        ranges = [*human_ranges, *robot_ranges, *plate_ranges]

        ids = []
        k = 1
        for r in ranges:
            idx = (full_id // k) % r
            ids.append(idx)
            k *= r

        if not humanonly:
            human_pos, human_hand_id, robot_pos, robot_hand_id, *plate_ids = ids
        else:
            human_pos, human_hand_id, *plate_ids = ids
            robot_pos = self.initial_state['robot_pos']
            robot_hand_id = 0


        human_hand_id -= 1
        if human_hand_id == -1:
            human_hand = ''
        else:
            human_hand = items[human_hand_id]

        robot_hand_id -= 1
        if robot_hand_id == -1:
            robot_hand = ''
        else:
            robot_hand = items[robot_hand_id]


        plate = tuple(item for item, idx in zip(items, plate_ids) if idx == 1)

        state = {
            'human_pos' : human_pos,
            'human_hand' : human_hand,
            'robot_pos' : robot_pos,
            'robot_hand' : robot_hand,
            'plate' : plate,
        }

        return state


    def get_num_states(self, humanonly=False):
        items = list(self.counter_items.values())[:-1]
        num_items = len(items)

        pos_range = self.num_pos
        hand_range = num_items + 1
        human_ranges = [pos_range, hand_range]

        if humanonly:
            robot_ranges = []
        else:
            robot_ranges = human_ranges.copy()

        plate_ranges = [2] * num_items

        ranges = [*human_ranges, *robot_ranges, *plate_ranges]
        return functools.reduce(lambda a, b : a * b, ranges)

    
    def get_T_and_Rs(self):
        print('Computing T...')
        nS = self.get_num_states()
        nAh = self.human_action_space.n
        nAr = self.robot_action_space.n
        nR = len(self.reward_distribution)


        T = {}
        T_shape = (nS, nAh, nAr, nS)
        Rs = [{} for _ in range(nR)]
        R_shape = (nS, nAh, nAr, nS)
        print(T_shape)

        for s_id, a_h, a_r in itertools.product(range(nS), range(nAh), range(nAr)):
            s = self.id_to_state(s_id)

            next_s = self.transition_fn(s, a_h, a_r)
            next_s_id = self.state_to_id(next_s)

            T[s_id, a_h, a_r, next_s_id] = 1.0
            for idx in range(nR):
                Rs[idx][s_id, a_h, a_r, next_s_id] = self.reward_fn(state=s, human_action=a_h, robot_action=a_r, next_state=next_s, reward_idx=idx)

            if len(T) % 10000 == 0:
                print(len(T))

        T = dict_to_sparse(T, T_shape)
        Rs = [dict_to_sparse(R, R_shape) for R in Rs]

        print('Computing T and Rs... Done!')
        return T, Rs

    def get_T(self):
        print('Computing T...')
        nS = self.get_num_states()
        nAh = self.human_action_space.n
        nAr = self.robot_action_space.n

        T = {}
        T_shape = (nS, nAh, nAr, nS)
        print(T_shape)

        for s_id, a_h, a_r in itertools.product(range(nS), range(nAh), range(nAr)):
            s = self.id_to_state(s_id)

            next_s = self.transition_fn(s, a_h, a_r)
            next_s_id = self.state_to_id(next_s)

            T[s_id, a_h, a_r, next_s_id] = 1.0

            if len(T) % 10000 == 0:
                print(len(T))

        T = dict_to_sparse(T, T_shape)

        print('Computing T... Done!')
        return T

    def get_T_humanonly(self):
        print('Computing T...')
        nS = self.get_num_states(humanonly=True)
        nA = self.human_action_space.n

        T = {}
        T_shape = (nS, nA, nS)

        for s_id, a in itertools.product(range(nS), range(nA)):
            s = self.id_to_state(s_id, humanonly=True)

            next_s = self.transition_fn(s, a)
            next_s_id = self.state_to_id(next_s, humanonly=True)

            T[s_id, a, next_s_id] = 1.0

        T = dict_to_sparse(T, T_shape)

        print('Computing T... Done!')
        return T

    
    def get_R(self, reward_idx):
        nS = self.get_num_states()
        nAh = self.human_action_space.n
        nAr = self.robot_action_space.n

        R = {}
        R_shape = (nS, nAh, nAr, nS)

        reward_fn = self.reward_distribution[reward_idx][0]

        for s_id, a_h, a_r in itertools.product(range(nS), range(nAh), range(nAr)):
            s = self.id_to_state(s_id)

            next_s = self.transition_fn(s, a_h, a_r)
            next_s_id = self.state_to_id(next_s)

            R[s_id, a_h, a_r, next_s_id] = reward_fn(state=s, human_action=a_h, robot_action=a_r, next_state=next_s)

            if len(R) % 10000 == 0:
                print(len(R))

        R = dict_to_sparse(R, R_shape)

        return R


    def get_R_humanonly(self, reward_idx):
        nS = self.get_num_states(humanonly=True)
        nA = self.human_action_space.n

        R = {}
        R_shape = (nS, nA, nS)

        reward_fn = self.reward_distribution[reward_idx][0]

        for s_id, a in itertools.product(range(nS), range(nA)):
            s = self.id_to_state(s_id, humanonly=True)

            next_s = self.transition_fn(s, a)
            next_s_id = self.state_to_id(next_s, humanonly=True)

            R[s_id, a, next_s_id] = reward_fn(state=s, human_action=a, next_state=next_s)

        R = dict_to_sparse(R, R_shape)

        return R


def get_pedagogical_T(ag, T, LH_P):
    nR, nS, nA = LH_P.shape




def compute_pedagogic_human_policy_fn(ag):        
    print('Computing pedagogic human policy...')
    # Compute LiteralHuman policies
    nS = ag.get_num_states(humanonly=True)
    nA = ag.human_action_space.n
    nR = len(ag.reward_distribution)

    LH_P = np.empty((nR, nS, nA))

    use_egreedy = False

    T = ag.get_T_humanonly()
    Rs = [ag.get_R_humanonly(r_idx) for r_idx in range(nR)]
    for r_idx, R in enumerate(Rs):
        # We make the reward state,action only, which
        # is fine here, since the env is deterministic
        R = R.sum(axis=2)
        num_iter = 1000
        LH_P[r_idx] = softhard_value_iteration(T, R, discount=ag.discount, beta=1e0, num_iter=num_iter)
        if use_egreedy:
            EPS = 5e-4
            LH_P += EPS / nA
            LH_P /= LH_P.sum(axis=2)[:, :, None]

    # EPS = 1e-3
    # LH_P[LH_P < EPS] = EPS
    # LH_P /= LH_P.sum(axis=2)[:, :, None]

    # LiteralRobot belief updates
    LR_B = LH_P / np.sum(LH_P, axis=0)

    ped_T = get_pedagogical_T(ag, LH_P)
    ped_Rs = get_pedagogical_Rs(ag, LH_P)

    for r_idx, R in enumerate(Rs):
        R_new = R.copy()

    # Compute PedagogicHuman policies
    PH_P = LR_B / np.sum(LR_B, axis=2)[:, :, None]

    print('Computing pedagogic human policy... Done!')
    for i in range(0, 12, 3):
        print(i, '\n', LH_P[:, i])
        print(i, '\n', LR_B[:, i])
        print(i, '\n', PH_P[:, i])
    breakpoint()

    def human_policy_fn(unused_ag, reward_idx):
        def human_policy(state):
            state_id = ag.state_to_id(state, humanonly=True)
            return sample_distribution(PH_P[reward_idx, state_id])
        return human_policy

    return human_policy_fn

class TabularMiniPieAssistanceGame(AssistanceGame):
    def __init__(self):
        f_ag = MiniPieGridworldAssistanceGame()

        num_states = f_ag.get_num_states()
        state_space = Discrete(num_states)
        human_action_space = f_ag.human_action_space
        robot_action_space = f_ag.robot_action_space
        # T = f_ag.get_T()
        T, Rs = f_ag.get_T_and_Rs()
        rewards_dist = [(R, prob) for R, (_, prob) in zip(Rs, f_ag.reward_distribution)]
        horizon = f_ag.horizon
        discount = f_ag.discount

        initial_state_id = f_ag.state_to_id(f_ag.initial_state)
        initial_state_dist = np.zeros(num_states)
        initial_state_dist[initial_state_id] = 1.0

        self.f_ag = f_ag

        super().__init__(
            state_space=state_space,
            human_action_space=human_action_space,
            robot_action_space=robot_action_space,
            transition=T,
            reward_distribution=rewards_dist,
            initial_state_distribution=initial_state_dist,
            horizon=horizon,
            discount=discount,
        )



class MiniPieGridworldAssistanceProblem(AssistanceProblem):
    def __init__(self, human_policy_fn=functional_random_policy_fn, **kwargs):
        assistance_game = MiniPieGridworldAssistanceGame()
        self.ag = assistance_game

        human_policy_fn = minipie_human_policy_fn
        # human_policy_fn = compute_pedagogic_human_policy_fn(self.ag)

        functional = True
        if functional:
            state_space_builder = pie_state_space_builder
            transition_model_fn_builder = pie_transition_model_fn_builder
            reward_model_fn_builder = pie_reward_model_fn_builder
            sensor_model_fn_builder = pie_sensor_model_fn_builder
            observation_model_fn = pie_observation_model_fn_builder(assistance_game)

            super().__init__(
                assistance_game=assistance_game,
                human_policy_fn=human_policy_fn,

                state_space_builder=state_space_builder,
                transition_model_fn_builder=transition_model_fn_builder,
                reward_model_fn_builder=reward_model_fn_builder,
                sensor_model_fn_builder=sensor_model_fn_builder,
                observation_model_fn=observation_model_fn,
            )

        else:
            self.t_ag = TabularMiniPieAssistanceGame()
            super().__init__(
                assistance_game=self.t_ag,
                human_policy_fn=human_policy_fn,
            )
            

    def render(self, mode='human'):
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
            'D' : partial(make_image_transform, 'cherry1.png'),
            'E' : partial(make_image_transform, 'chocolate4.png', c=(0.1, 0.1, 0.1)),
            'F' : partial(make_image_transform, 'apple3.png', c=(0.7, 0.3, 0.2)),
            'P' : partial(make_image_transform, 'plate1.png', w=1.3, h=1.3),
            '+' : partial(make_image_transform, 'plus1.png', w=0.5, h=0.5),
            '=' : partial(make_image_transform, 'equal1.png', w=0.5, h=0.2),
            '0' : partial(make_image_transform, 'pie1.png'),
            '1' : partial(make_image_transform, 'cake1.png'),
            '2' : partial(make_image_transform, 'apple-pie2.png', c=(0.7, 0.3, 0.2)),
        }

        def move_to_counter(pos):
            if not isinstance(pos, tuple):
                pos = poss[pos]

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

        poss = [
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),

            (1, 3),

            (2, 3),
            (2, 2),
            (2, 1),
            (2, 0),
        ]

        g_x0 = -110 + 1.5*grid_side
        g_y0 = -110 + 1*grid_side

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

                make_grid_rect(1, 0, 1, height-1),

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



        if isinstance(self.state, dict):
            state = self.state
        else:
            nS0 = self.t_ag.state_space.n
            idx = self.state % nS0
            rew_idx = self.state // nS0
            state = self.ag.id_to_state(self.state)
            state['reward_idx'] = rew_idx

        print(state)

        human_pos = state['human_pos']
        robot_pos = state['robot_pos']
        human_hand = state['human_hand']
        robot_hand = state['robot_hand']
        plate = state['plate']
        reward_idx = state['reward_idx']
        recipe_made = state['recipe_made']

        def coords_from_pos(pos):
            if not isinstance(pos, tuple):
                pos = poss[pos]
            return self.grid.coords_from_pos(pos)


        human_coords = coords_from_pos(human_pos)
        self.human_transform.set_translation(*human_coords)

        robot_coords = coords_from_pos(robot_pos)
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

        if recipe_made:
            for idx, recipe in enumerate(self.ag.recipes):
                if plate == recipe:
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
            return one_hot(pos, ag.num_pos)

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

    # pos_vec_len = 2 * ag.width + 2 * ag.height
    pos_vec_len = 2 * ag.num_pos

    num_dims = pos_vec_len + 3 * num_ingredients
    low = np.zeros(num_dims)
    high = np.ones(num_dims)
    high[:4] = [ag.width, ag.height, ag.width, ag.height]

    ob_space = Box(low=low, high=high)

    observation_model_fn = partial(FunctionalObservationModel, fn=observation_fn, space=ob_space)
    return observation_model_fn


def get_minipie_hardcoded_robot_policy(env, *args, **kwargs):
    class Policy:
        def predict(self, ob, state=None):
            t, r_idx = state if state is not None else (0, None)

            num_pos = env.ag.num_pos

            onehot_pos = ob[:num_pos]
            onehot_y = ob[num_pos:2*num_pos]

            human_pos = np.argmax(onehot_pos)

            if t == 2:
                if human_pos == 3:
                    r_idx = 1
                else:
                    r_idx = 0

            # S, R, U, L, D, A = range(6)
            R, L, A = range(3)


            robot_policies = [
                [
                    R, R, # Wait 1 step
                    L, L, L, A, # Get dark flour
                    L, A,    # Take to plate
                ],

                [
                    R, R,             # Wait 1 step
                    A,                # get milk chocolate
                    L, L, L, L, A, # drop in plate
                    R, A,             # get dark flour
                    L, A,             # Take to plate
                ],

                [
                    L, L, L, A, # Get dark flour
                    L, A,    # Take to plate
                    R, R, R, R, A,                # get milk chocolate
                    L, L, L, L, A, # drop in plate
                ],
            ]

            # robot_policies[1:] = reversed(robot_policies[1:])

            if r_idx is None:
                robot_policy = robot_policies[0]
            else:
                robot_policy = robot_policies[r_idx]

            act = robot_policy[t] if t < len(robot_policy) else A

            return act, (t+1, r_idx)

    return Policy()


def minipie_human_policy_fn(ag, reward):
    def human_policy(state):
        R, L, A = range(3)

        policy0 = [
            R, A,          # get white flour
            R, R, A, # take to plate
            A,          # bake
        ]

        trail0 = [
            ((0, 1), '', {'A' : 0}),
            ((0, 2), '', {'A' : 0}),

            ((0, 2), 'A', {'A' : 0}),
            ((0, 3), 'A', {'A' : 0}),
            ((1, 3), 'A', {'A' : 0}),

            ((1, 3), '', {'A' : 1}),
        ]


        policy1 = [
            R, R, A, # get green apple
            R, A, # drop in plate
            L, L, A, # get white flour
            R, R, A, # take to plate
            A, # bake
        ]

        trail1 = [
            ((0, 1), '',  {'C' : 0}),
            ((0, 2), '',  {'C' : 0}),
            ((0, 3), '',  {'C' : 0}),
            ((0, 3), 'C', {'C' : 0}),
            ((1, 3), 'C', {'C' : 0}),

            ((1, 3), '', {'A' : 0, 'C' : 1}),
            ((0, 3), '', {'A' : 0, 'C' : 1}),
            ((0, 2), '', {'A' : 0, 'C' : 1}),

            ((0, 2), 'A', {'A' : 0, 'C' : 1}),
            ((0, 3), 'A', {'A' : 0, 'C' : 1}),
            ((1, 3), 'A', {'A' : 0, 'C' : 1}),

            ((1, 3), '', {'A' : 1, 'C' : 1}),
        ]

        reward_idx = state['reward_idx']

        policy = (policy0, policy1)[reward_idx]
        trail = (trail0, trail1)[reward_idx]


        def has_plate_condition(plate, cond):
            c = Counter(plate)
            return all(c[k] == v for k, v in cond.items())

        def get_time(state):
            for t, (pos, hand, plate_cond) in enumerate(trail):
                if (
                    pos == ag.poss[state['human_pos']] and
                    hand == state['human_hand'] and
                    has_plate_condition(state['plate'], plate_cond)
                ):
                    return t

        t = get_time(state)
        action = policy[t] if t is not None else A

        return action

    return human_policy

def tabular_minipie_human_policy_fn(ag, reward, reward_idx):
    def func_human_policy(state_id):
        R, L, A = range(3)

        policy0 = [
            R, A,          # get white flour
            R, R, A, # take to plate
            A,          # bake
        ]

        trail0 = [
            ((0, 1), '', {'A' : 0}),
            ((0, 2), '', {'A' : 0}),

            ((0, 2), 'A', {'A' : 0}),
            ((0, 3), 'A', {'A' : 0}),
            ((1, 3), 'A', {'A' : 0}),

            ((1, 3), '', {'A' : 1}),
        ]


        policy1 = [
            R, R, A, # get green apple
            R, A, # drop in plate
            L, L, A, # get white flour
            R, R, A, # take to plate
            A, # bake
        ]

        trail1 = [
            ((0, 1), '',  {'C' : 0}),
            ((0, 2), '',  {'C' : 0}),
            ((0, 3), '',  {'C' : 0}),
            ((0, 3), 'C', {'C' : 0}),
            ((1, 3), 'C', {'C' : 0}),

            ((1, 3), '', {'A' : 0, 'C' : 1}),
            ((0, 3), '', {'A' : 0, 'C' : 1}),
            ((0, 2), '', {'A' : 0, 'C' : 1}),

            ((0, 2), 'A', {'A' : 0, 'C' : 1}),
            ((0, 3), 'A', {'A' : 0, 'C' : 1}),
            ((1, 3), 'A', {'A' : 0, 'C' : 1}),

            ((1, 3), '', {'A' : 1, 'C' : 1}),
        ]

        policy = (policy0, policy1)[reward_idx]
        trail = (trail0, trail1)[reward_idx]

        state = ag.f_ag.id_to_state(state_id)

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

    nS, nAh, *_ = ag.transition.shape

    human_policy = np.zeros((nS, nAh))

    for state_id in range(nS):
        act = func_human_policy(state_id)
        human_policy[state_id, act] = 1.0

    return human_policy
