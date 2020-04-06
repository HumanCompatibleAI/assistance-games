"""Core classes, such as POMDP and AssistanceGame.
"""

import numpy as np
import gym
from gym.spaces import Discrete, Box

from assistance_games.utils import sample_distribution, uniform_simplex_sample


class POMDP(gym.Env):
    def __init__(
        self,
        state_space,
        sensor_space,
        action_space,
        transition,
        sensor,
        rewards,
        initial_state_distribution,
        initial_belief=None,
        horizon=None,
        back_sensor=None,
        discount=1.0,
    ):
        """Partially Observable Markov Decision Process environment.

        Parameters
        ----------
        state_space : gym.spaces.Discrete, S
        sensor_space : gym.spaces.Discrete, O
        action_space : gym.spaces.Discrete, A
        transition : np.array[|S|, |A|, |S|]
        rewards : np.array[|S|, |A|, |S|]
        sensor : np.array[|A|, |S|, |O|]
        back_sensor : np.array[|A|, |S|, |O'|]
        initial_state_distribution : np.array[|S|]
        initial_belief : np.array[|S|]
        horizon : Float
        discount : Float
        """
        if initial_belief is None:
            initial_belief = initial_state_distribution

        self.state_space = state_space
        self.sensor_space = sensor_space
        self.action_space = action_space
        self.transition = transition
        self.sensor = sensor
        self.back_sensor = back_sensor
        self.rewards = rewards
        self.horizon = horizon
        self.initial_state_distribution = initial_state_distribution
        self.initial_belief = initial_belief
        self.discount = discount
        self.viewer = None

        self.belief_space = Box(low=0.0, high=1.0, shape=(state_space.n,))

    def reset(self):
        self.state = sample_distribution(self.initial_state_distribution)
        self.t = 0
        self.belief = self.initial_belief
        return self.belief

    def step(self, act):
        assert act in self.action_space

        old_state = self.state
        self.state = sample_distribution(self.transition[self.state, act])

        old_belief = self.belief
        ob = self.sample_obs(act, self.state)
        self.belief = self.update_belief(self.belief, act, ob)

        # Observed reward is myopic
        observed_reward = old_belief @ self.rewards[:, act, :] @ self.belief
        true_reward = self.rewards[old_state, act, self.state]

        self.t += 1
        done = self.horizon is not None and self.t >= self.horizon

        info = {'ob' : ob, 'true_reward' : true_reward}

        return self.belief, observed_reward, done, info
        
    def render(self):
        print(self.state)

    def sample_obs(self, act, state):
        return sample_distribution(self.sensor[act, state])

    def update_belief(self, belief, act, ob):
        new_belief = (belief @ self.transition[:, act, :]) * self.sensor[act, :, ob]
        new_belief /= new_belief.sum()
        return new_belief

    @property
    def observation_space(self):
        return self.belief_space


class AssistanceGame:
    def __init__(
        self,
        state_space,
        human_action_space,
        robot_action_space,
        transition,
        reward_distribution,
        initial_state_distribution,
        horizon=None,
        discount=1.0,
    ):
        """Two-agent MDP, with shared reward hidden from second agent.

        Parameters
        ----------
        state_space : gym.spaces.Discrete, S
        human_action_space : gym.spaces.Discrete, A_h
        robot_action_space : gym.spaces.Discrete, A_r
        transition : np.array[|S|, |A_h|, |A_r|, |S|]
        reward_distribution : List[Tuple[np.array[|S|, |A_h|, |A_r|, |S|], Float]]
        initial_state_distribution : np.array[|S|]
        horizon : Float
        discount : Float
        """
        self.state_space = state_space
        self.human_action_space = human_action_space
        self.robot_action_space = robot_action_space
        self.transition = transition
        self.reward_distribution = reward_distribution
        self.initial_state_distribution = initial_state_distribution
        self.horizon = horizon
        self.discount = discount


class AssistanceProblem(POMDP):
    def __init__(self, assistance_game, human_policy_fn):
        """
        Parameters
        ----------
        assistance_game : AssistanceGame
        human_policy_fn : AssistanceGame -> Reward (np.array[|S|, |A_h|, |A_r|, |S|])
                                         -> Policy (np.array[|S|, |A|])
        """
        ag = assistance_game

        sensor_space = ag.state_space

        action_space = ag.robot_action_space
        num_actions = action_space.n

        num_rewards = len(ag.reward_distribution)
        num_states = ag.state_space.n * num_rewards
        state_space = Discrete(num_states)

        sensor = np.zeros((num_actions, num_states, ag.state_space.n))
        back_sensor = np.zeros((num_actions, num_states, ag.human_action_space.n))
        transition = np.zeros((num_states, num_actions, num_states))
        rewards = np.zeros((num_states, action_space.n, num_states))

        for reward_idx, (reward, _) in enumerate(ag.reward_distribution): 
            human_policy = human_policy_fn(assistance_game, reward)
            for ag_state in range(ag.state_space.n):
                state = ag.state_space.n * reward_idx + ag_state

                sensor[:, state, ag_state] = 1.0
                back_sensor[:, state] = human_policy[ag_state]

                for ag_next_state in range(ag.state_space.n):
                    next_state = ag.state_space.n * reward_idx + ag_next_state

                    transition[state, :, next_state] = human_policy[ag_state] @ ag.transition[ag_state, :, :, ag_next_state]
                    rewards[state, :, next_state] = human_policy[ag_state] @ reward[ag_state, :, :, ag_next_state]

        reward_distribution = np.array([prob for _, prob in ag.reward_distribution])
        initial_state_distribution = reward_distribution.reshape(-1, 1) @ ag.initial_state_distribution.reshape(1, -1)
        initial_state_distribution = initial_state_distribution.reshape(-1)

        discount = ag.discount
        horizon = ag.horizon

        super().__init__(
            state_space=state_space,
            sensor_space=sensor_space,
            action_space=action_space,
            transition=transition,
            sensor=sensor,
            back_sensor=back_sensor,
            rewards=rewards,
            horizon=horizon,
            initial_state_distribution=initial_state_distribution,
            discount=discount,
        )
        self.num_obs = ag.state_space.n
        self.num_rewards = num_rewards


class POMDPPolicy:
    """Policy from alpha vectors provided by POMDP solvers"""
    def __init__(self, alphas):
        self.alpha_vectors = []
        self.alpha_actions = []
        for vec, act in alphas:
            self.alpha_vectors.append(vec)
            self.alpha_actions.append(act)

    def predict(self, belief, state=None, deterministic=True):
        idx = np.argmax(self.alpha_vectors @ belief)
        return self.alpha_actions[idx], state


### Human Policies

def random_policy_fn(assistance_game, reward):
    num_states = assistance_game.state_space.n
    num_actions = assistance_game.human_action_space.n
    return np.full((num_states, num_actions), 1 / num_actions)


def hard_value_iteration(assistance_game, reward):
    ag = assistance_game

    # We want to learn a time independent policy here,
    # so that we get time independent transitions in
    # our assistance problem.
    # So we assume/force the game to be infinite horizon
    # and discounted.
    # This should not be an issue for most environments.
    discount = min(0.9, ag.discount)
    num_iter = 50

    num_states = ag.state_space.n
    num_actions = ag.human_action_space.n

    # Robot model - we assume here that
    # the robot acts randomly
    transition = ag.transition.mean(axis=2)
    reward = reward.mean(axis=2)

    Q = np.empty((num_states, num_actions))
    V = np.zeros(num_states)
    for t in range(num_iter):
        for s in range(num_states):
            for a in range(num_actions):
                Q[s, a] = transition[s, a, :] @ (reward[s, a, :] + discount * V)
        V = np.max(Q, axis=1)

    policy = np.eye(num_actions)[Q.argmax(axis=1)]
    return policy

