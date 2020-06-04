from math import ceil
from pathlib import Path
import os

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines.results_plotter import load_results, ts2xy, plot_results, X_TIMESTEPS

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    cumm = [values[0]]
    for v in values[1:]:
        cumm.append(cumm[-1] + v)

    avgs = []
    for i, v in enumerate(cumm):
        if i - window >= 0:
            avg = (cumm[i] - cumm[i - window]) / window
        else:
            avg = cumm[i] / (i + 1)

        avgs.append(avg)

    return avgs


def smooth_arr(arr, alpha=0.07):
    new_arr = [arr[0]]
    for value in arr[1:]:
        new_value = (1 - alpha) * new_arr[-1] + alpha * value
        new_arr.append(new_value)
    return new_arr


def plot_results_several_seeds_in_one_file(log_folder='./logs', steps_per_seed=1e6, window=100, expert_return=None, title='Learning Curves'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x_all, y_all = ts2xy(load_results(log_folder), 'timesteps')

    if x_all[-1] >= steps_per_seed:
        pts_per_seed = np.where(x_all > steps_per_seed)[0][0] 
        n_seeds = ceil((len(x_all) - 20) / pts_per_seed)
    else:
        pts_per_seed = len(x_all)
        n_seeds = 1
    print('#seeds = {}'.format(n_seeds))
    fig = plt.figure(title, figsize = (10, 4))
    for i in range(n_seeds):
        x = x_all[i*pts_per_seed:(i+1)*pts_per_seed]
        y = y_all[i*pts_per_seed:(i+1)*pts_per_seed]
        y_smoothed = moving_average(y, window=window)
        # Truncate x
        x = x[len(x) - len(y_smoothed):] - min(x)
        y = y[len(y) - len(y_smoothed):]
        plt.plot(x, y_smoothed)
    if expert_return is not None:
        plt.axhline(y=expert_return, color='k', linestyle='.')

    plt.xlabel('Number of Timesteps')
    plt.ylabel('Total reward')
    plt.title(title + " (Smoothed)")
    plt.show()


def plot_multiple_runs1(log_folder, window=100, expert_return=None, legend=True, title='Learning Curves'):
    fig = plt.figure(title, figsize = (10, 4))
    paths = Path(log_folder).iterdir()
    for path in paths:
        x, y = ts2xy(load_results(path), 'timesteps')
        y_smoothed = moving_average(y, window=window)
        plt.plot(x, y_smoothed, label='seed ' + str(path)[-1])
    if legend:
        plt.legend()
    if expert_return is not None:
        plt.axhline(y=expert_return, color='k', linestyle='--')
    plt.xlabel('Timesteps')
    plt.ylabel('Total reward')
    plt.title(title + " (Smoothed)")
    plt.show()

def plot_multiple_runs2(log_folder, window=100, expert_return=None, legend=True, title='Learning Curves'):
    fig = plt.figure(title, figsize = (10, 4))
    for path in Path(log_folder).glob('**/*'):
        if path.suffix != '.npz':
            continue

        res = np.load(path, allow_pickle=True)
        x, y = res['timesteps'], res['results']
        y = y.mean(axis=1).flatten()
        y_smoothed = moving_average(y, window=window)
        plt.plot(x, y_smoothed, label='seed ' + str(path)[-1])
    if legend:
        plt.legend()
    if expert_return is not None:
        plt.axhline(y=expert_return, color='k', linestyle='--')
    plt.xlabel('Timesteps')
    plt.ylabel('Total reward')
    plt.title(title + " (Smoothed)")
    plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', type=str, default='./logs')
    parser.add_argument('-a', '--title', type=str, default='Learning curves')
    parser.add_argument('-w', '--window', type=int, default=10000)
    args = parser.parse_args()

    
    Path(args.directory).mkdir(parents=True, exist_ok=True)
    assert Path(args.directory).exists()
    plot_multiple_runs2(args.directory, legend=False, window=args.window, expert_return=12.77)


if __name__ == '__main__':
    main()
