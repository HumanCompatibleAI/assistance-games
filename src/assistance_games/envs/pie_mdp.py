from functools import partial
import itertools

from gym.spaces import Discrete
import numpy as np

from assistance_games.core import (
    AssistanceGame,
    AssistanceProblem,
    get_human_policy,
    BeliefObservationModel,
    DiscreteFeatureSenseObservationModel,
    discrete_reward_model_fn_builder,
)

from assistance_games.utils import get_asset


class PieMDPGame(AssistanceGame):
    def __init__(self):
        human_state_space = Discrete(5)
        human_action_space = Discrete(3)

        robot_state_space = Discrete(4)
        robot_action_space = Discrete(2)

        nSh = human_state_space.n
        nAh = human_action_space.n
        nSr = robot_state_space.n
        nAr = robot_action_space.n

        nS = nSh * nSr
        state_space = Discrete(nS)

        T = np.zeros((nSh, nSr, nAh, nAr, nSh, nSr))

        R0 = np.zeros((nSh, nSr, nAh, nAr, nSh, nSr))
        R1 = np.zeros((nSh, nSr, nAh, nAr, nSh, nSr))

        for s_h, s_r, a_h, a_r in itertools.product(range(nSh), range(nSr), range(nAh), range(nAr)):
            n_s_h, n_s_r = self.transition_fn(s_h, s_r, a_h, a_r)
            T[s_h, s_r, a_h, a_r, n_s_h, n_s_r] = 1.0
            
        R0[0, :, 2, :, :, :] = -1.0
        R1[0, :, 2, :, :, :] = -1.0

        R0[:, 1, :, :, :, 2] = 10.0
        R0[:, 1, :, :, :, 3] = 1.0

        R1[:, 1, :, :, :, 2] = 1.0
        R1[:, 1, :, :, :, 3] = 10.0

        T = T.reshape(nS, nAh, nAr, nS)
        R0 = R0.reshape(nS, nAh, nAr, nS)
        R1 = R1.reshape(nS, nAh, nAr, nS)


        initial_state_dist = np.zeros(nS)
        initial_state_dist[0] = 1.0

        rewards_dist = [(R0, 0.5), (R1, 0.5)]

        horizon = 5
        discount = 1.0

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

    @staticmethod
    def transition_fn(s_h, s_r, a_h, a_r):
        if s_h == 0:
            n_s_h = a_h + 1
        elif s_h == 3:
            n_s_h = 4
        else:
            n_s_h = s_h

        if s_r == 0 and n_s_h in (1, 2, 4):
            n_s_r = 1
        elif s_r == 1:
            n_s_r = 1 + a_r + 1
        else:
            n_s_r = s_r

        return n_s_h, n_s_r


def get_pie_mdp_expert(assistance_game, reward, **kwargs):
    is_R0 = reward[1, 0, 0, 2] > 2.0

    P = 1/3 * np.ones(reward.shape[:2])
    if is_R0:
        P[0, 0] = 0.5
        P[0, 1] = 0.5
        P[0, 2] = 0.0
    else:
        P[0, 0] = 0.0
        P[0, 1] = 0.0
        P[0, 2] = 1.0
    
    return P


class PieMDPAssistanceProblem(AssistanceProblem):
    def __init__(self, human_policy_fn=get_pie_mdp_expert, use_belief_space=True):
        assistance_game = PieMDPGame()

        if use_belief_space:
            observation_model_fn = BeliefObservationModel
        else:
            feature_extractor = lambda state : state % assistance_game.state_space.n
            setattr(feature_extractor, 'n', assistance_game.state_space.n)
            observation_model_fn = partial(DiscreteFeatureSenseObservationModel, feature_extractor=feature_extractor)

        reward_model_fn_builder = partial(discrete_reward_model_fn_builder, use_belief_space=use_belief_space)

        super().__init__(
            assistance_game=assistance_game,
            human_policy_fn=human_policy_fn,
            observation_model_fn=observation_model_fn,
            reward_model_fn_builder=reward_model_fn_builder,
        )
        self.ag_state_space_n = assistance_game.state_space.n


    def render(self, mode='human'):
        import assistance_games.rendering as rendering

        if self.viewer is None:
            self.viewer = rendering.Viewer(500,600)
            self.viewer.set_bounds(-120, 120, -150, 120)

            grid_side = 30

            self.grid = rendering.Grid(start=(-110, -110), grid_side=grid_side, shape=(7, 6))
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

            robot_coords = self.grid.coords_from_pos((4, 3))
            self.robot_transform.set_translation(*robot_coords)


            make_rect = lambda x, y, w, h : rendering.make_polygon([(x,y),(x+w,y),(x+w,y+h),(x,y+h)])
            gs = grid_side
            make_grid_rect = lambda i, j, di, dj : make_rect(-110 + i*gs, -110 + j*gs, di*gs, dj*gs)

            # Top counter  
            counters = [
                make_grid_rect(0, 0, 1, 6),
                make_grid_rect(3, 0, 1, 6),
                make_grid_rect(6, 0, 1, 6),
                make_grid_rect(0, 0, 7, 1),
                make_grid_rect(0, 5, 7, 1),
            ]
            for counter in counters:
                r = 0.8
                off = 0.05
                g = r - off
                b = r - 2 * off
                counter.set_color(r, g, b)
                self.viewer.add_geom(counter)

            flour_image = get_asset('images/flour3.png')
            scale = 0.6
            flour = rendering.Image(flour_image, scale * grid_side, scale * grid_side)
            flour_transform = rendering.Transform()
            flour.add_attr(flour_transform)
            self.viewer.add_geom(flour)

            flour_coords = self.grid.coords_from_pos((0, 3))
            flour_transform.set_translation(*flour_coords)


            apple_image = get_asset('images/apple3.png')
            scale = 0.6
            apple = rendering.Image(apple_image, scale * grid_side, scale * grid_side)
            apple.set_color(0.5, 0.7, 0.0)
            apple_transform = rendering.Transform()
            apple.add_attr(apple_transform)
            self.viewer.add_geom(apple)

            apple_coords = self.grid.coords_from_pos((2, 5))
            apple_transform.set_translation(*apple_coords)


            chocolate_image = get_asset('images/chocolate2.png')
            scale = 0.7
            chocolate = rendering.Image(chocolate_image, scale * grid_side, scale * grid_side)
            chocolate_transform = rendering.Transform()
            chocolate.add_attr(chocolate_transform)
            self.viewer.add_geom(chocolate)

            chocolate_coords = self.grid.coords_from_pos((2, 0))
            chocolate_transform.set_translation(*chocolate_coords)


            applepie_image = get_asset('images/apple-pie2.png')
            scale = 0.7
            applepie = rendering.Image(applepie_image, scale * grid_side, scale * grid_side)
            applepie_transform = rendering.Transform()
            applepie.add_attr(applepie_transform)

            applepie_coords = self.grid.coords_from_pos((3, 3))
            applepie_transform.set_translation(*applepie_coords)


            chocpie_image = get_asset('images/chocolate-pie2.png')
            scale = 0.7
            chocpie = rendering.Image(chocpie_image, scale * grid_side, scale * grid_side)
            chocpie_transform = rendering.Transform()
            chocpie.add_attr(chocpie_transform)

            chocpie_coords = self.grid.coords_from_pos((3, 3))
            chocpie_transform.set_translation(*chocpie_coords)

            self.applepie = applepie
            self.chocpie = chocpie




            hl = 15
            header_x = -15 + hl
            header_y = -110 + 6 * grid_side + hl

            scale = 0.4
            flour2 = rendering.Image(flour_image, scale * grid_side, scale * grid_side)
            flour2_transform = rendering.Transform()
            flour2.add_attr(flour2_transform)
            self.viewer.add_geom(flour2)

            flour2_transform.set_translation(header_x, header_y)


            plus_image = get_asset('images/plus1.png')

            scale = 0.2
            plus1 = rendering.Image(plus_image, scale * grid_side, scale * grid_side)
            plus1_transform = rendering.Transform()
            plus1.add_attr(plus1_transform)
            self.viewer.add_geom(plus1)

            plus1_transform.set_translation(header_x + 1*hl, header_y)

            scale = 0.2
            plus2 = rendering.Image(plus_image, scale * grid_side, scale * grid_side)
            plus2_transform = rendering.Transform()
            plus2.add_attr(plus2_transform)
            self.viewer.add_geom(plus2)

            plus2_transform.set_translation(header_x + 3*hl, header_y)

            scale = 0.2
            plus2 = rendering.Image(plus_image, scale * grid_side, scale * grid_side)
            plus2_transform = rendering.Transform()
            plus2.add_attr(plus2_transform)
            self.viewer.add_geom(plus2)

            plus2_transform.set_translation(header_x + 1*hl, header_y + 1.2*hl)



            equal_image = get_asset('images/equal1.png')

            scale = 0.15
            equal1 = rendering.Image(equal_image, scale * grid_side, scale * grid_side)
            equal1_transform = rendering.Transform()
            equal1.add_attr(equal1_transform)
            self.viewer.add_geom(equal1)

            equal1_transform.set_translation(header_x + 5*hl, header_y)

            scale = 0.15
            equal2 = rendering.Image(equal_image, scale * grid_side, scale * grid_side)
            equal2_transform = rendering.Transform()
            equal2.add_attr(equal2_transform)
            self.viewer.add_geom(equal2)

            equal2_transform.set_translation(header_x + 3*hl, header_y + 1.2*hl)



            scale = 0.4
            apple2 = rendering.Image(apple_image, scale * grid_side, scale * grid_side)
            apple2.set_color(0.5, 0.7, 0.0)
            apple2_transform = rendering.Transform()
            apple2.add_attr(apple2_transform)
            self.viewer.add_geom(apple2)

            apple2_transform.set_translation(header_x + 2*hl, header_y)


            scale = 0.4
            chocolate2 = rendering.Image(chocolate_image, scale * grid_side, scale * grid_side)
            chocolate2_transform = rendering.Transform()
            chocolate2.add_attr(chocolate2_transform)
            self.viewer.add_geom(chocolate2)

            chocolate2_transform.set_translation(header_x + 4*hl, header_y)



            scale = 0.4
            flour3 = rendering.Image(flour_image, scale * grid_side, scale * grid_side)
            flour2_transform = rendering.Transform()
            flour3.add_attr(flour2_transform)
            self.viewer.add_geom(flour3)

            flour2_transform.set_translation(header_x, header_y + 1.2*hl)


            scale = 0.4
            apple3 = rendering.Image(apple_image, scale * grid_side, scale * grid_side)
            apple3.set_color(0.5, 0.7, 0.0)
            apple3_transform = rendering.Transform()
            apple3.add_attr(apple3_transform)
            self.viewer.add_geom(apple3)

            apple3_transform.set_translation(header_x + 2*hl, header_y + 1.2*hl)


            scale = 0.4
            applepie2 = rendering.Image(applepie_image, scale * grid_side, scale * grid_side)
            applepie2_transform = rendering.Transform()
            applepie2.add_attr(applepie2_transform)
            self.viewer.add_geom(applepie2)

            applepie2_transform.set_translation(header_x + 4*hl, header_y + 1.2*hl)


            scale = 0.4
            chocpie2 = rendering.Image(chocpie_image, scale * grid_side, scale * grid_side)
            chocpie2_transform = rendering.Transform()
            chocpie2.add_attr(chocpie2_transform)
            self.viewer.add_geom(chocpie2)

            chocpie2_transform.set_translation(header_x + 6*hl, header_y)


            scale = 1.0
            thought = rendering.make_ellipse(scale * grid_side/2, scale * 0.7*grid_side/2)
            thought.set_color(0.9, 0.9, 0.9)
            self.thought_transform = rendering.Transform()
            thought.add_attr(self.thought_transform)

            self.viewer.add_geom(thought)

            scale = 0.17
            thought2 = rendering.make_ellipse(scale * grid_side/2, scale * grid_side/2)
            thought2.set_color(0.9, 0.9, 0.9)
            self.thought_transform2 = rendering.Transform()
            thought2.add_attr(self.thought_transform2)
            self.viewer.add_geom(thought2)

            scale = 0.1
            thought3 = rendering.make_ellipse(scale * grid_side/2, scale * grid_side/2)
            thought3.set_color(0.9, 0.9, 0.9)
            self.thought_transform3 = rendering.Transform()
            thought3.add_attr(self.thought_transform3)
            self.viewer.add_geom(thought3)



            hl2 = 20
            header2_x = -110 + hl2
            header2_y = -110 + 6 * grid_side + hl2

            scale = 0.3
            applepie3 = rendering.Image(applepie_image, scale * grid_side, scale * grid_side)
            applepie3_transform = rendering.Transform()
            applepie3.add_attr(applepie3_transform)
            self.viewer.add_geom(applepie3)

            applepie3_transform.set_translation(header2_x, header2_y)
            self.tgt_apple_transform = applepie3_transform

            scale = 0.3
            chocpie3 = rendering.Image(chocpie_image, scale * grid_side, scale * grid_side)
            chocpie3_transform = rendering.Transform()
            chocpie3.add_attr(chocpie3_transform)
            self.viewer.add_geom(chocpie3)

            chocpie3_transform.set_translation(header2_x + 2*hl2, header2_y)
            self.tgt_choc_transform = chocpie3_transform


            comp_transform = rendering.Transform()
            self.comp_transform = comp_transform

            greater_image = get_asset('images/greater1.png')

            scale = 0.15
            greater1 = rendering.Image(greater_image, scale * grid_side, scale * grid_side)
            greater1.add_attr(comp_transform)

            less_image = get_asset('images/less1.png')

            scale = 0.15
            less1 = rendering.Image(less_image, scale * grid_side, scale * grid_side)
            less1.add_attr(comp_transform)


            self.greater = greater1
            self.less = less1




        human_grid_pos = [
            (2, 3),
            (1, 3),
            (2, 4),
            (2, 2),
            (2, 1),
        ]

        reward_idx = self.state // self.ag_state_space_n
        state = self.state % self.ag_state_space_n

        human_state = state // 4
        robot_state = state % 4


        human_pos = human_grid_pos[human_state]

        human_coords = self.grid.coords_from_pos(human_pos)
        self.human_transform.set_translation(*human_coords)

        thought_pos = (lambda x, y : (x-1,y+1))(*human_pos)
        thought_coords = self.grid.coords_from_pos(thought_pos)
        self.thought_transform.set_translation(*thought_coords)
        thought_coords2 = (lambda x, y : (x+12,y-12))(*thought_coords)
        self.thought_transform2.set_translation(*thought_coords2)
        thought_coords3 = (lambda x, y : (x+17,y-17))(*thought_coords)
        self.thought_transform3.set_translation(*thought_coords3)

        tgt_apple_coords = (lambda x, y : (x-8,y))(*thought_coords)
        tgt_comp_coords = thought_coords
        tgt_choc_coords = (lambda x, y : (x+8,y))(*thought_coords)

        self.tgt_apple_transform.set_translation(*tgt_apple_coords)
        self.comp_transform.set_translation(*tgt_comp_coords)
        self.tgt_choc_transform.set_translation(*tgt_choc_coords)

        if robot_state == 2:
            self.viewer.add_onetime(self.applepie)
        elif robot_state == 3:
            self.viewer.add_onetime(self.chocpie)

        if reward_idx == 0:
            self.viewer.add_onetime(self.greater)
        else:
            self.viewer.add_onetime(self.less)


        print(reward_idx, human_state, robot_state)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
