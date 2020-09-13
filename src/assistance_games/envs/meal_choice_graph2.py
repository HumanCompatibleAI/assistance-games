from collections import namedtuple
from gym.spaces import Discrete, Box
import numpy as np

import assistance_games.rendering as rendering
from assistance_games.utils import get_asset, MOVEMENT_ACTIONS
from assistance_games.core.core2 import AssistancePOMDPWithMatrixSupport, UniformDiscreteDistribution, KroneckerDistribution


MealChoiceState = namedtuple('MealChoiceState', ['world', 'time'])


class MealChoice2(AssistancePOMDPWithMatrixSupport):
    """An environment in which R must ask H about their preferences after H returns.

    A state consists of the world state and the current time. The
    robot has to make either cake or pizza, but doesn't know which H
    prefers, and H is currently at work. R knows when H will return,
    and must use this information to figure out whether to guess at
    H's preference, or to wait for H to ask about their preferences.
    """
    WORLD_STATES = ['flour', 'dough', 'cake', 'pizza', 'end']
    ROBOT_ACTIONS = ['noop', 'create1', 'create2', 'query']
    HUMAN_ACTIONS = ['noop', 'cake', 'pizza']
    THETAS = ['cake', 'pizza']

    WORLD_STATE_TO_INDEX = {v:i for i, v in enumerate(WORLD_STATES)}
    ROBOT_ACTION_TO_INDEX = {v:i for i, v in enumerate(ROBOT_ACTIONS)}
    HUMAN_ACTION_TO_INDEX = {v:i for i, v in enumerate(HUMAN_ACTIONS)}

    def __init__(self, time_when_feedback_available=3, horizon=6):
        self.num_world_states = len(MealChoice2.WORLD_STATES)
        self.nS = len(MealChoice2.WORLD_STATES) * (horizon + 1)
        self.nAH = len(MealChoice2.HUMAN_ACTIONS)
        self.nAR = len(MealChoice2.ROBOT_ACTIONS)
        self.nOR = self.nS  # Fully observable
        self.time_when_feedback_available = time_when_feedback_available
        # One-hot world state, one-hot aH vector, and time
        self.num_features = self.num_world_states + 4
        init_state = MealChoiceState(world='flour', time=0)

        super().__init__(
            discount=0.9,
            horizon=horizon,
            theta_dist=UniformDiscreteDistribution(MealChoice2.THETAS),
            init_state_dist=KroneckerDistribution(init_state),
            observation_space=Box(
                low=np.array([0] * self.num_features),
                high=np.array([1] * (self.num_features - 1) + [horizon])
            ),
            action_space=Discrete(len(MealChoice2.ROBOT_ACTIONS)),
            default_aH='noop',
            default_aR='noop',
            deterministic=True,
            fully_observable=True
        )

    def state_to_index(self, state):
        world_idx = MealChoice2.WORLD_STATE_TO_INDEX[state.world]
        return state.time * self.num_world_states + world_idx

    def index_to_state(self, num):
        time, world_idx = num // self.num_world_states, num % self.num_world_states
        world = MealChoice2.WORLD_STATES[world_idx]
        return MealChoiceState(world=world, time=time)

    def human_action_to_index(self, aH):
        return MealChoice2.HUMAN_ACTION_TO_INDEX[aH]

    def index_to_human_action(self, num):
        return MealChoice2.HUMAN_ACTIONS[num]

    def robot_action_to_index(self, aR):
        return MealChoice2.ROBOT_ACTION_TO_INDEX[aR]

    def index_to_robot_action(self, num):
        return MealChoice2.ROBOT_ACTIONS[num]

    def encode_obs(self, obs, prev_aH):
        world_idx = MealChoice2.WORLD_STATE_TO_INDEX[obs.world]
        prev_aH_idx = self.human_action_to_index(prev_aH)

        result = np.zeros(self.num_features)
        result[world_idx] = 1
        result[self.num_world_states + prev_aH_idx] = 1
        result[-1] = obs.time
        return result

    def decode_obs(self, encoded_obs):
        world_idx = np.argmax(encoded_obs[:self.num_world_states])
        prev_aH_idx = np.argmax(encoded_obs[self.num_world_states:-1])
        time = int(encoded_obs[-1])

        world = MealChoice2.WORLD_STATES[world_idx]
        prev_aH = self.index_to_human_action(prev_aH_idx)
        return MealChoiceState(world=world, time=time), prev_aH

    def get_transition_distribution(self, state, aH, aR):
        if self.is_terminal(state):
            return KroneckerDistribution(state)

        # aH is already passed to R, and when aR = query is already
        # passed to H, so we don't have to handle either of those
        if aR in ['noop', 'query']:
            next_world = state.world
        elif state.world == 'dough':
            next_world = 'cake' if aR == 'create1' else 'pizza'
        else:
            next_world = {
                'flour': 'dough',
                'cake': 'end',
                'pizza': 'end',
                'end': 'end'
            }[state.world]

        new_state = MealChoiceState(world=next_world, time=state.time+1)
        return KroneckerDistribution(new_state)


    def get_reward(self, state, aH, aR, next_state, theta):
        if next_state.world == 'end' and state.world != 'end':
            if state.world == theta:
                return 2  # Made the right thing
            else:
                return -1  # Made the wrong thing
        return 0

    def get_human_action_distribution(self, obsH, prev_aR, theta):
        h_returned = obsH.time >= self.time_when_feedback_available
        robot_asked_query = (prev_aR == 'query')
        return KroneckerDistribution(theta if h_returned and robot_asked_query else 'noop')

    def is_terminal(self, state):
        return state.time >= self.horizon

    def close(self):
        # if self.viewer is not None:
        #     self.viewer.close()
        return super().close()

    def render(self, state, prev_aH, prev_aR, theta, mode='human'):
        if prev_aR != None:
            print('aH = {}, aR = {}'.format(prev_aH, prev_aR))
        print('s = {}, t = {}, human wants {}'.format(state.world, state.time, theta))


def get_meal_choice_hardcoded_robot_policy(*args, **kwargs):
    class Policy:
        def predict(self, ob, state=None):
            N, C1, C2, Q = MealChoice2.ROBOT_ACTIONS

            # Hacky way of detecting a reset
            if state is None:
                self.actions = [C1, N, Q, N, 'ob', C1]

            aR = self.actions.pop(0)
            if aR != 'ob':
                return aR, 'ignored'
            else:
                assert ob[6] == 1 or ob[7] == 1
                return (C1 if ob[6] == 1 else C2), 'ignored'

    return Policy()
