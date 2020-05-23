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

from collections import namedtuple
import functools
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from scipy.special import softmax
from scipy.spatial import distance_matrix

import cvxpy as cp

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
    max_value_iter=10,
    limit_belief_expansion=True,
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
        R = pomdp.reward_model.R
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

    def find_domination_witness(alphas, target, unpacked_constraints=False):
        """Finds witness for target vector.

        That is, finds a belief for which the target alpha
        is works better than all the other ones, proving that
        the target alpha is not dominated.
        """
        if not alphas:
            # With no alphas, any belief is a witness
            belief = np.zeros_like(target.vector)
            belief[0] = 1.0
            return belief

        nAL = len(alphas)
        nS = len(target.vector)
        EPS = 1e-3

        belief = cp.Variable(nS)
        slack = cp.Variable(1)
        
        obj = cp.Maximize(slack)

        # These two branches are meant to be the same; however, the upper
        # branch is much easier to understand, while the lower one is more
        # efficient (often 10-100 times faster). So, I'm keeping the
        # upper one for readability and testing purposes.
        if unpacked_constraints:
            target_is_not_dominated = [(target.vector - a.vector) @ belief >= slack for a in alphas]
            belief_is_prob = [np.ones(num_states) @ belief == 1, 0 <= belief, belief <= 1]
            slack_is_positive = [slack >= EPS]
            constraints = target_is_not_dominated + belief_is_prob + slack_is_positive
        else:
            A_not_dominated = np.zeros((nAL, nS+1))
            for A_row, alpha in zip(A_not_dominated, alphas):
                A_row[:-1] = target.vector - alpha.vector
                A_row[-1] = -1 # slack coefficient
            b_not_dominated = np.zeros((nAL))

            A_belief_bounded_by_one = np.zeros((nS, nS+1))
            A_belief_bounded_by_one[:, :nS] = np.eye(nS)
            b_belief_bounded_by_one = np.ones((nS))

            A_belief_bounded_by_zero = np.zeros((nS, nS+1))
            A_belief_bounded_by_zero[:, :nS] = np.eye(nS)
            b_belief_bounded_by_zero = np.zeros((nS))

            A_belief_sums_to_one = np.zeros((1, nS+1))
            A_belief_sums_to_one[0, :nS] = np.ones((nS))
            b_belief_sums_to_one = np.ones((1,))

            A_slack_positive = np.zeros((1, nS+1))
            A_slack_positive[0, -1] = 1.0
            b_slack_positive = EPS * np.ones((1,))

            A_ub = np.concatenate([
                (-1) * A_not_dominated,
                (-1) * A_belief_bounded_by_zero,
                A_belief_bounded_by_one,
                (-1) * A_slack_positive,
            ])

            b_ub = np.concatenate([
                (-1) * b_not_dominated,
                (-1) * b_belief_bounded_by_zero,
                b_belief_bounded_by_one,
                (-1) * b_slack_positive,
            ])

            A_eq = A_belief_sums_to_one
            b_eq = b_belief_sums_to_one

            c = np.zeros(nS + 1)
            c[-1] = -1.0

            constraints = [A_ub[:, :-1] @ belief + A_ub[:, -1:] @ slack <= b_ub,
                           A_eq[:, :-1] @ belief + A_eq[:, -1:] @ slack == b_eq]

        prob = cp.Problem(obj, constraints)

        # It seems that it is common for LP solvers to not guarantee a solution,
        # and just throw an exception if they don't find a solution, we try to
        # use other solvers, which most often works.
        try:
            prob.solve(solver=cp.ECOS, verbose=False)
            return belief.value
        except cp.error.SolverError as err:
            pass

        try:
            prob.solve(solver=cp.SCS, verbose=False)
            return belief.value
        except cp.error.SolverError:
            pass

        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)
            return result.x[:-1] if result.success else None
        except Exception as err:
            pass

        raise Exception(
            'All LP solvers failed. You may want to try slightly alter values in your problem, or ' +
            'try different solver hyperparameters (or different solvers) by directly modifying ' +
            'the code.'
        )


    successes = []
    queue = alpha_pairs

    while queue:
        alpha_set = successes + queue
        belief = find_domination_witness(alpha_set[:-1], alpha_set[-1])
        if belief is not None:
            successes.append(queue[-1])
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


def deep_rl_solve(pomdp, total_timesteps=100000, learning_rate=1e-3, use_lstm=True):
    from stable_baselines import PPO2
    from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy

    if use_lstm:
        policy = PPO2(MlpLstmPolicy, pomdp, learning_rate=learning_rate, nminibatches=1, policy_kwargs=dict(n_lstm=32))
    else:
        policy = PPO2(MlpPolicy, pomdp, learning_rate=learning_rate)
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
