from argparse import ArgumentParser
from functools import partial
import numpy as np
import time

from .deep_rl import dqn_solve, ppo_solve
from .classic_solvers import pbvi, exact_vi
import assistance_games.envs as envs
from .core import (
    ReducedAssistancePOMDP,
    ReducedAssistancePOMDPWithMatrices,
    ReducedFullyObservableDeterministicAssistancePOMDPWithMatrices
)


def run_environment(env, discount, policy=None, num_episodes=10, dt=0.01, max_steps=100, render_mode='human'):
    if num_episodes == -1:
        num_episodes = int(5e6)

    def render_fn(prev_action=None):
        if render_mode is not None:
            if prev_action:
                result = env.render(mode=render_mode, prev_action=prev_action)
            else:
                result = env.render(mode=render_mode)
            
            time.sleep(dt)
            return result

    def log(s):
        if render_mode is not None:
            print(s)

    render_results = []
    rewards, discounted_rewards = [], []
    for ep in range(num_episodes):
        log('\n starting ep {}'.format(ep))
        total_reward, total_discounted_reward, cur_discount = 0, 0, 1
        ob = env.reset()
        render_results.append([])
        render_results[ep].append(render_fn())

        state = None
        done = False
        step = 0
        while not done and step < max_steps:
            if policy is None:
                ac = env.action_space.sample()
            else:
                ac, state = policy.predict(ob, state)
            ob, re, done, _ = env.step(ac)

            log('r = {}'.format(re))
            log('ac = {}'.format(ac))
            total_reward += re
            total_discounted_reward += cur_discount * re
            cur_discount *= discount
            render_results[ep].append(render_fn(ac))
            step += 1
        rewards.append(total_reward)
        discounted_rewards.append(total_discounted_reward)

    log('Undiscounted rewards: {}'.format(rewards))
    print('Average undiscounted reward: {}'.format(sum(rewards) / len(rewards)))
    log('Discounted rewards: {}'.format(discounted_rewards))
    print('Average discounted reward: {}'.format(sum(discounted_rewards) / len(discounted_rewards)))
    return render_results


def get_env_fn(env_name):
    name_to_env_fn = {
        'cake_or_pie': envs.CakeOrPieGridworld,
        'mealchoice': envs.MealChoice,
        'pie_small': envs.SmallPieGridworld,
        'redblue': envs.RedBlue,
        'wardrobe': envs.Wardrobe,
        'worms': envs.WormyApples,
    }
    return name_to_env_fn[env_name]


def get_hardcoded_policy(env, env_name, *args, **kwargs):
    hardcoded_policies = {
        'cake_or_pie': envs.get_cake_or_pie_hardcoded_robot_policy,
        'mealchoice': envs.get_meal_choice_hardcoded_robot_policy,
        'pie_small': envs.get_small_pie_hardcoded_robot_policy,
    }
    if env_name not in hardcoded_policies:
        raise ValueError("No hardcoded robot policy for this environment.")
    return hardcoded_policies[env_name](env, *args, **kwargs)


# Doesn't always work, since it requires that there are only 255 colors in each frame
def save_results_to_gif(results, filename, fps=5, end_of_trajectory_pause=3):
    from array2gif import write_gif
    dataset = []
    for episode in results:
        for frame in episode:
            dataset.append(frame)
        for _ in range(end_of_trajectory_pause):
            dataset.append(frame)
    write_gif(dataset, filename, fps=fps)


def run(env_name, env_kwargs, algo_name, seed=0, render=True, num_episodes=10, **kwargs):
    algos = {
        'dqn' : dqn_solve,
        'exact' : exact_vi,
        'hardcoded' : partial(get_hardcoded_policy, env_name=env_name),
        'pbvi' : pbvi,
        'ppo' : ppo_solve,
        'random' : lambda *args, **kwargs : None,
    }
    algo = algos[algo_name]
    env = get_env_fn(env_name)(**env_kwargs)
    discount = env.discount
    if algo_name not in ('exact', 'pbvi'):
        env = ReducedAssistancePOMDP(env)
    elif env.fully_observable and env.deterministic:
        env = ReducedFullyObservableDeterministicAssistancePOMDPWithMatrices(env)
    else:
        env = ReducedAssistancePOMDPWithMatrices(env)
    
    print('\n Running algorithm {} with seed {}'.format(algo_name, seed))
    np.random.seed(seed)
    policy = algo(env, seed=seed, **kwargs)
    if render:
        results = run_environment(env, discount, policy, dt=0.5, num_episodes=num_episodes, render_mode='human')
        # TODO saving to gif doesn't work yet
        # save_results_to_gif(results, filename=f'{env_name}_{algo_name}_s{seed}.gif')
    env.close()


def str_to_dict(s):
    """Converts a string to a dictionary, used for specifying keyword args on
    the command line. We try converting values to bool, int and float; if none
    of these work we leave the values as strings."""
    if s == '':
        return {}

    def convert(val):
        if val == 'True': return True
        if val == 'False': return False
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        return val

    kvs = s.split(',')
    kv_pairs = [elem.split(':') for elem in kvs]
    result = {x: convert(y) for x, y in kv_pairs}
    print(result)
    return result


def main():
    parser = ArgumentParser(description="Minimal script to solve and render an environment.")
    parser.add_argument('-a', '--algo_name', type=str, default='pbvi')
    parser.add_argument('-e', '--env_name', type=str, default='redblue')
    parser.add_argument('-k', '--env_kwargs', type=str, default='')
    parser.add_argument('-l', '--log_dir', default='assistance-logs', type=str)
    parser.add_argument('-ln', '--log_name', type=str, help="Name of the TensorBoard run")
    parser.add_argument('-m', '--num_runs', type=int, default=1)
    parser.add_argument('-n', '--total_timesteps', type=int, default=25_000_000)
    parser.add_argument('-nl', '--no-log', action='store_true')
    parser.add_argument('-nr', '--no_render', dest='render', action='store_false')
    parser.add_argument('-o', '--output_folder', type=str, default='')
    parser.add_argument('-p', '--num_episodes', type=int, default=10)
    parser.add_argument('-r', '--render', default=True, action='store_true')
    parser.add_argument('-s', '--seed', type=int, default=0)
    args = parser.parse_args()

    for run_id in range(args.num_runs):
        seed = args.seed + run_id   # Use a different seed for each run

        if args.log_name is None:
            base_name = f"{args.env_name}_{args.algo_name}"
            if args.env_kwargs:
                base_name += f"_{args.env_kwargs}"
        else:
            base_name = args.log_name
        
        run(
            env_name=args.env_name,
            env_kwargs=str_to_dict(args.env_kwargs),
            algo_name=args.algo_name,
            seed=seed,
            total_timesteps=args.total_timesteps,
            render=args.render,
            num_episodes=args.num_episodes,
            log_dir=args.log_dir if not args.no_log else None,
            log_name=f"{base_name}_seed{args.seed}",
        )


if __name__ == '__main__':
    main()
