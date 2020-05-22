"""POMDP solvers.

Include exact plan-based solvers, approximate plan-based solvers,
and deep rl solvers.

*Some resources for understanding POMDP solvers*
- Slides with the intuition for solvers:
    https://www.cs.cmu.edu/~ggordon/780-fall07/lectures/POMDP_lecture.pdf
- PBVI paper:
    http://www.cs.cmu.edu/~ggordon/jpineau-ggordon-thrun.ijcai03.pdf
- Survey of point-based solvers, has clearest presentation:
    https://www.cs.mcgill.ca/~jpineau/files/jpineau-jaamas12-finalcopy.pdf
"""
import pathlib
from collections import namedtuple
import functools
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from scipy.special import softmax
from scipy.spatial import distance_matrix

from assistance_games.core import POMDPPolicy, TabularBackwardSensorModel
from assistance_games.utils import sample_distribution, uniform_simplex_sample, force_dense

Alpha = namedtuple('Alpha', ['vector', 'action'])

def pomdp_value_iteration(
    pomdp,
    *,
    expand_beliefs_fn,
    value_backup_fn,
    max_iter=2,
    num_beliefs=30,
    max_value_iter=20,
    limit_belief_expansion=True
):
    """Value Iteration POMDP solver.

    Implements different exact or approximate solvers, differing on how
    beliefs are selected (if any), and how alpha values are updated.
    """
    num_value_iter = min(max_value_iter, get_effective_horizon(pomdp))
    nS = pomdp.state_space.n

    beliefs = None
    alphas = [Alpha(np.zeros(nS), None)]
    for _ in range(max_iter):
        beliefs = expand_beliefs_fn(pomdp, beliefs, num_beliefs, limit_belief_expansion=limit_belief_expansion)
        for _ in range(num_value_iter):
            alphas = value_backup_fn(pomdp, alphas, beliefs)

    return POMDPPolicy(alphas)

def get_effective_horizon(pomdp, epsilon=1e-3):
    """Effective horizon for the POMDP.

    For infinite horizon POMDPs, returns h such
    that (R_max - R_min) * discount**h < epsilon.
    """
    if pomdp.horizon is not None:
        return pomdp.horizon
    else:
        disc = pomdp.discount
        R = pomdp.rewards
        R_range = R.max() - R.min()
        num_iters = int(np.ceil(np.log(epsilon / R_range) // np.log(disc)))
        return num_iters


def none_expand_beliefs_fn(*args, **kwargs):
    return None


def pbvi_expand_beliefs_fn(pomdp, beliefs=None, num_beliefs=30, limit_belief_expansion=True):
    if beliefs is None:
        return sample_random_beliefs(pomdp, num_beliefs)
    else:
        max_new_beliefs = num_beliefs if limit_belief_expansion else None
        return density_expand_beliefs(pomdp, beliefs, max_new_beliefs)

def sample_random_beliefs(pomdp, num_beliefs):
    """Mixed sampling for beliefs.

    Half of the points are uniformly sampled from simplex;
    the other half by sampling a beta in [0, 4], and applying 
    softmax to a point uniformly sampled from [0, 1]**n.
    """
    num_states = pomdp.state_space.n
    logits = np.random.randn(num_beliefs // 2, num_states)
    betas = 4 * np.random.rand(num_beliefs // 2, 1)
    beliefs = softmax(betas * logits, axis=1)
    unif = np.stack([uniform_simplex_sample(num_states) for _ in range(num_beliefs - num_beliefs // 2)])
    beliefs = np.concatenate([beliefs, unif], axis=0)
    return beliefs

def density_expand_beliefs(pomdp, beliefs, max_new_beliefs=None, epsilon=1e-2):
    """Expand belief set by sampling next beliefs.

    For each belief, samples a next belief for each action,
    and then takes the point furthest from the belief set.

    If max_new_beliefs is None, then len(new_beliefs) <= 2 * len(beliefs).
    If max_new_beliefs is not None, then len(new_beliefs) <= max_new_beliefs + len(beliefs).
    """
    num_beliefs = len(beliefs)
    nA = pomdp.action_space.n
    T = pomdp.transition_model.T
    new_beliefs = list(beliefs)
    for i in np.random.permutation(num_beliefs)[:max_new_beliefs]:
        belief = beliefs[i]
        candidates = []
        for action in range(nA):
            state = sample_distribution(belief)
            next_state = sample_distribution(T[state, action])
            ob = pomdp.sensor_model.sample_sense(action=action, state=state, next_state=next_state)
            new_belief = pomdp.sensor_model.update_belief(belief, action, ob)
            candidates.append(new_belief)

        dists = distance_matrix(candidates, new_beliefs).min(axis=1)
        idx = np.argmax(dists)
        dist = dists[idx]

        if dist > epsilon:
            new_beliefs.append(candidates[idx])

    return new_beliefs


def exact_value_backup(pomdp, alphas, *args, **kwargs):
    """Performs exact value backup.

    This consists of the following phases:
    1 - Calculate new alpha vectors after taking each action.
        i - Calculate the contribution for each observation.
        ii - Compute the new alphas for each action and observation |-> alpha_i mapping.
    2 - Prune alphas that are dominated (i.e. not used for any possible belief).
    """
    nA = pomdp.action_space.n

    T = pomdp.transition_model.T
    R = pomdp.reward_model.R
    R = force_dense(np.sum(R * T, axis=2))

    obs_alphas = compute_obs_alphas(pomdp, alphas)

    new_alphas = []
    for act in range(nA):
        new_alpha_vecs = cross_sums(obs_alphas[act], R[:, act])
        new_alphas.extend([Alpha(vector, act) for vector in new_alpha_vecs])

    new_alphas.extend(alphas)
    new_alphas = prune_alphas(new_alphas)
    return new_alphas

def cross_sums(V, v0):
    """Returns (v0 + sum(V[i][p[i]] for i in range(n)) for p in permutations(n))."""
    if V.shape[0] == 0:
        yield v0
    else:
        for vec in cross_sums(V[1:], v0):
            for val in V[0]:
                yield vec + val

def prune_alphas(alpha_pairs):
    """Removes alphas not used for any possible belief.

    For each alpha a_j, we check whether there exists
    a witness belief b (by solving a linear program)
    such that a_j @ b > a_i @ b, for all i, i != j.
    """
    def find_domination_witness(alphas, target):
        """Finds witness for target vector.
        """
        nAL = len(alphas)
        nS = len(target.vector)

        if not alphas:
            belief = np.zeros(nS)
            belief[0] = 1.0
            return belief

        A_ub = np.zeros((nAL + nS + 1, nS + 1))
        for A_row, (alp, _) in zip(A_ub[:nAL], alphas):
            A_row[:-1] = alp - target.vector
            A_row[-1] = 1

        A_ub[nAL:] = (-1) * np.eye(nS+1)

        A_eq = np.ones((1, nS + 1))
        A_eq[0, -1] = 0
        
        b_eq = np.ones(1)

        b = np.zeros(A_ub.shape[0])
        b[-1] = (-1) * 1e-4
        c = np.zeros(nS + 1)
        c[-1] = -1.0

        result = linprog(c, A_ub=A_ub, b_ub=b, A_eq=A_eq, b_eq=b_eq)
        if result.success:
            belief = result.x[:-1]
            return belief
        else:
            return None

    successes = []
    queue = alpha_pairs

    while queue:
        alpha_set = successes + queue
        belief = find_domination_witness(alpha_set[:-1], alpha_set[-1])
        if belief is not None:
            # We save the best alpha for the witness belief found.
            idx = max(range(len(queue)), key=lambda i : belief @ queue[i].vector)
            successes.append(queue[idx])
            if idx != len(queue) - 1:
                queue.pop(idx)
        queue.pop(-1)

    return successes


def compute_obs_alphas(pomdp, alphas):
    nS = pomdp.state_space.n
    nA = pomdp.action_space.n
    disc = pomdp.discount
    T = pomdp.transition_model.T

    use_back_sensor = isinstance(pomdp.sensor_model, TabularBackwardSensorModel)
    O = pomdp.sensor_model.sensor if not use_back_sensor else pomdp.sensor_model.back_sensor
    nO = O.shape[-1]

    alpha_vecs = [alpha.vector for alpha in alphas]

    obs_alphas = np.empty((nA, nO, len(alpha_vecs), nS))
    alpha_matrix = np.array(alpha_vecs).T
    for a in range(nA):
        for o in range(nO):
            if use_back_sensor:
                obs_alphas[a, o] = disc * (O[a, :, o] * (T[:, a] @ alpha_matrix).T)
            else:
                obs_alphas[a, o] = disc * (T[:, a] @ (O[a, :, o, None] * alpha_matrix)).T
    return obs_alphas



def point_based_value_backup(pomdp, alphas, beliefs=None, use_back_sensor=False):
    """Performs approximate value backup by tracking a finite set of beliefs.

    This consists of the following phases:
    1 - Calculate new alpha vectors after taking each action.
        i - Calculate the contribution for each observation.
        ii - Compute the new alphas for each action and belief, by selecting the next plan
             for each observation using the belief b.
    2 - For each belief, select the alpha that maximizes expected reward.
    """
    nS = pomdp.state_space.n
    nA = pomdp.action_space.n
    T = pomdp.transition_model.T

    use_back_sensor = isinstance(pomdp.sensor_model, TabularBackwardSensorModel)
    O = pomdp.sensor_model.sensor if not use_back_sensor else pomdp.sensor_model.back_sensor
    nO = O.shape[-1]

    R = pomdp.reward_model.R
    R = force_dense(np.sum(R * T, axis=2))

    def compute_action_alphas(obs_alphas, beliefs):
        action_alphas = np.empty((len(beliefs), nA, nS))
        for j, b in enumerate(beliefs):
            for a in range(nA):
                action_alphas[j, a] = R[:, a]
                for o in range(nO):
                    action_alphas[j, a] += max(obs_alphas[a, o], key=lambda alpha : alpha @ b)
        return action_alphas

    def select_best_alphas(action_alphas, beliefs):
        alpha_pairs = []
        for j, b in enumerate(beliefs):
            act = np.argmax(action_alphas[j] @ b)
            alpha_pairs.append(Alpha(action_alphas[j, act], act))
        return alpha_pairs
    
    obs_alphas = compute_obs_alphas(pomdp, alphas)
    action_alphas = compute_action_alphas(obs_alphas, beliefs)
    new_alphas = select_best_alphas(action_alphas, beliefs)

    return new_alphas


pbvi = functools.partial(
    pomdp_value_iteration,
    expand_beliefs_fn=pbvi_expand_beliefs_fn,
    value_backup_fn=point_based_value_backup,
)
exact_vi = functools.partial(
    pomdp_value_iteration,
    expand_beliefs_fn=none_expand_beliefs_fn,
    value_backup_fn=exact_value_backup,
)


def deep_rl_solve(pomdp, total_timesteps=1000000, learning_rate=1e-3, use_lstm=True, seed=0):
    from stable_baselines import PPO2
    from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy

    if use_lstm:
        policy = PPO2(MlpLstmPolicy,
                      pomdp,
                      learning_rate=learning_rate,
                      nminibatches=1,
                      policy_kwargs=dict(n_lstm=32),
                      ent_coef=0.011,
                      n_steps=256,
                      seed=seed,
                      tensorboard_log='./logs')
    else:
        policy = PPO2(MlpPolicy, pomdp, learning_rate=learning_rate, seed=seed)
    policy.learn(total_timesteps=total_timesteps)
    return policy

def get_venv(env, n_envs=1):
    """Simple wrapper to avoid importing stable-baselines and tensorflow when unnecessary.
    """
    from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
    if n_envs == 1:
        return DummyVecEnv([lambda : env])
    else:
        return SubprocVecEnv([(lambda : env) for _ in range(n_envs)])
