from collections import namedtuple
from functools import partial

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

import assistance_games.rendering as rendering
from assistance_games.utils import get_asset, dict_to_sparse


class WardrobeAssistanceGame(AssistanceGame):
    """

    This is the map of the assistance game:

    .......
    .     .
    .     .
    . TT  .
    . TT  .
    .H   R.
    .......

    H = Human
    R = Robot
    T = Wardrobe

    The human wants to move the wardrobe to one of the corners.
    To move the wardrobe, two agents need to push it at the same time
    in the same direction.

    Size of the map and of the table can be chosen.
    """

    State = namedtuple('State', ['human', 'robot', 'wardrobe'])

    def __init__(self, size=3, wardrobe_size=1):
        self.size = size
        self.wardrobe_size = wardrobe_size

        wardrobe_range = self.size - self.wardrobe_size + 1
        self.wardrobe_range = wardrobe_range

        target_and_reward =  {
            (0, 0) : {},
            (wardrobe_range - 1, 0) : {},
            (0, wardrobe_range - 1) : {},
            (wardrobe_range - 1, wardrobe_range - 1) : {},
        }
        self.targets = target_and_reward

        num_states = self.size ** 4 * self.wardrobe_range ** 2
        state_space = Discrete(num_states)

        num_actions = 4
        action_space = Discrete(num_actions)


        T = {}
        T_shape = (num_states, num_actions, num_actions, num_states)

        for hy in range(self.size):
            for hx in range(self.size):
                for ry in range(self.size):
                    for rx in range(self.size):
                        for ty in range(wardrobe_range):
                            for tx in range(wardrobe_range):
                                state = self.State(human=(hx, hy), robot=(rx, ry), wardrobe=(tx, ty))
                                idx = self.get_idx(state)
                                for ah in range(num_actions):
                                    for ar in range(num_actions):
                                        next_state = self.transition_fn(state, ah, ar)
                                        next_idx = self.get_idx(next_state)
                                        T[idx, ah, ar, next_idx] = 1.0
                                        if next_state.wardrobe in target_and_reward and state.wardrobe != next_state.wardrobe:
                                            target_and_reward[next_state.wardrobe][idx, ah, ar, next_idx] = 1.0

        transition = dict_to_sparse(T, T_shape)

        rewards_dist = [(dict_to_sparse(R, T_shape), 0.25) for R in target_and_reward.values()]

        initial_state = ((0, 0), (self.size - 1, 0), (1, 1))
        initial_state_dist = np.zeros(num_states)
        initial_state_dist[self.get_idx(initial_state)] = 1.0


        horizon = 3 * self.size
        discount = 0.7

        super().__init__(
            state_space=state_space,
            human_action_space=action_space,
            robot_action_space=action_space,
            transition=transition,
            reward_distribution=rewards_dist,
            initial_state_distribution=initial_state_dist,
            horizon=horizon,
            discount=discount,
        )


    def transition_fn(self, state, human_action, robot_action):
        next_human_pos = self.move(state.human, human_action)
        next_robot_pos = self.move(state.robot, robot_action)

        if (human_action == robot_action and
            self.wardrobe_can_move(state.wardrobe, human_action) and
            self.in_wardrobe(state, next_human_pos) and
            self.in_wardrobe(state, next_robot_pos)
        ):
            next_wardrobe_pos = self.move(state.wardrobe, human_action)
        else:
            next_wardrobe_pos = state.wardrobe
            next_human_pos = next_human_pos if not self.in_wardrobe(state, next_human_pos) else state.human
            next_robot_pos = next_robot_pos if not self.in_wardrobe(state, next_robot_pos) else state.robot

        return self.State(human=next_human_pos,
                          robot=next_robot_pos,
                          wardrobe=next_wardrobe_pos)


    def move(self, pos, act, size=None):
        if size is None:
            size = self.size

        x, y = pos
        dirs = [
            (1, 0),
            (0, 1),
            (-1, 0),
            (0, -1),
        ]
        dx, dy = dirs[act]

        new_x = np.clip(x + dx, 0, size - 1)
        new_y = np.clip(y + dy, 0, size - 1)

        return new_x, new_y

    def wardrobe_can_move(self, pos, act):
        return self.move(pos, act, size=self.wardrobe_range) != pos


    def in_wardrobe(self, state, pos):
        wardrobe_x, wardrobe_y = state.wardrobe
        x, y = pos
        return (0 <= x - wardrobe_x < self.wardrobe_size and
                0 <= y - wardrobe_y < self.wardrobe_size)

    def get_idx(self, state):
        (hx, hy), (rx, ry), (tx, ty) = state

        return (
            ty + self.wardrobe_range * (
            tx + self.wardrobe_range * (
            ry + self.size * (
            rx + self.size * (
            hy + self.size * (
            hx)))))
        )

    def get_state(self, idx):
        steps = [self.wardrobe_range, self.wardrobe_range, self.size, self.size, self.size, self.size]

        vals = []
        for step in steps:
            vals.append(idx % step)
            idx //= step

        hx, hy, rx, ry, tx, ty = reversed(vals)
        return self.State(human=(hx, hy), robot=(rx, ry), wardrobe=(tx, ty))


class WardrobeAssistanceProblem(AssistanceProblem):
    def __init__(self, human_policy_fn=get_human_policy, use_belief_space=True):
        self.assistance_game = WardrobeAssistanceGame()

        if use_belief_space:
            observation_model_fn = BeliefObservationModel
        else:
            feature_extractor = lambda state : state % self.assistance_game.state_space.n
            setattr(feature_extractor, 'n', self.assistance_game.state_space.n)
            observation_model_fn = partial(DiscreteFeatureSenseObservationModel, feature_extractor=feature_extractor)

        reward_model_fn_builder = partial(discrete_reward_model_fn_builder, use_belief_space=use_belief_space)

        super().__init__(
            assistance_game=self.assistance_game,
            human_policy_fn=human_policy_fn,
            observation_model_fn=observation_model_fn,
            reward_model_fn_builder=reward_model_fn_builder,
        )

    def render(self, mode='human'):
        size = self.assistance_game.size
        wardrobe_size = self.assistance_game.wardrobe_size

        if self.viewer is None:
            self.viewer = rendering.Viewer(500,600)
            self.viewer.set_bounds(-120, 120, -150, 120)

            self.grid = rendering.Grid(start=(-100, -100), end=(100, 100), shape=(size, size))
            self.viewer.add_geom(self.grid)

            human_image = get_asset('images/girl1.png')
            human = rendering.Image(human_image, self.grid.side, self.grid.side)
            self.human_transform = rendering.Transform()
            human.add_attr(self.human_transform)
            self.viewer.add_geom(human)

            robot_image = get_asset('images/robot1.png')
            robot = rendering.Image(robot_image, self.grid.side, self.grid.side)
            self.robot_transform = rendering.Transform()
            robot.add_attr(self.robot_transform)
            self.viewer.add_geom(robot)

            wardrobe_image = get_asset('images/wardrobe1.png')
            wardrobe = rendering.Image(wardrobe_image, wardrobe_size * self.grid.side, wardrobe_size * self.grid.side)
            self.wardrobe_transform = rendering.Transform()
            wardrobe.add_attr(self.wardrobe_transform)
            self.viewer.add_geom(wardrobe)

        nS0 = self.assistance_game.state_space.n
        idx = self.state % nS0
        rew_idx = self.state // nS0

        human_pos, robot_pos, wardrobe_top_left = self.assistance_game.get_state(idx)

        human_coords = self.grid.coords_from_pos(human_pos)
        robot_coords = self.grid.coords_from_pos(robot_pos)

        tl_x, tl_y = wardrobe_top_left
        wardrobe_bot_right = tl_x + wardrobe_size - 1, tl_y + wardrobe_size - 1

        tl_x, tl_y = self.grid.coords_from_pos(wardrobe_top_left)
        br_x, br_y = self.grid.coords_from_pos(wardrobe_bot_right)

        wardrobe_coords = ((tl_x + br_x) / 2, (tl_y + br_y) / 2)

        self.human_transform.set_translation(*human_coords)
        self.robot_transform.set_translation(*robot_coords)
        self.wardrobe_transform.set_translation(*wardrobe_coords)


        def add_bar(pos, ratio):
            x, y = self.grid.coords_from_pos(pos)
            xs, ys = self.grid.x_step, self.grid.y_step
            l, r = x - xs/2, x - xs/2 + 3
            b = y - ys/2
            t = b + ratio * ys
            bar = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            bar.set_color(0.7, 0.3, 0.3)
            self.viewer.add_onetime(bar)

        if hasattr(self.observation_model, 'belief'):
            reward_beliefs = self.observation_model.belief.reshape(-1, nS0).sum(axis=1)

            for pos, ratio in zip(self.assistance_game.targets, reward_beliefs):
                add_bar(pos, ratio)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
