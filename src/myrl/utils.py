import time
from functools import wraps
import csv
import logging
import os
from math import ceil

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


logger = logging.getLogger(__name__)


def visualize(csv_path, out_path=None, window=301, title=None):
    if out_path is None:
        out_path = os.path.splitext(csv_path)[0] + '.png'
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    df.set_index(df.columns[0], inplace=True)
    rolled = df.rolling(window=window, center=True)
    median_df = rolled.median()
    quantile25_df = rolled.quantile(0.25)
    quantile75_df = rolled.quantile(0.75)

    ncol = 2
    nrow = ceil(len(median_df.columns) / ncol)

    index = median_df.index
    fig = plt.figure(figsize=(8 * ncol, 3 * nrow))
    axes = fig.subplots(nrows=nrow, ncols=ncol, sharex=True)
    for idx, col in enumerate(median_df.columns):
        ax = axes[idx // ncol][idx % ncol]
        ax.plot(index, median_df[col], alpha=0.8)
        ax.fill_between(index, quantile25_df[col], quantile75_df[col], alpha=0.3)
        ax.grid(True)
        ax.set_title(col)
        if (idx // ncol) == (nrow - 1):  # bottom ax
            ax.set_xlabel(index.name)

    ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())

    fig = ax.figure
    if title is None:
        title = ''
    fig.suptitle(title)
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    fig.savefig(out_path)
    logger.info(f'create figure at {out_path}')
    plt.close(fig)


def visualize_multi(root_path, out_path=None, window=301, title=None):
    """root_path: results/UpNDownNoFrameskip-v4  とか"""
    if out_path is None:
        out_path = os.path.join(root_path, 'history.png')
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)

    median_dfs = {}
    quantile25_dfs = {}
    quantile75_dfs = {}
    agent_names = [d for d in os.listdir(root_path) if not d.startswith('.')]
    for agent_name in agent_names:
        tmp_median_dfs = []
        tmp_quantile25_dfs = []
        tmp_quantile75_dfs = []
        attempts = [d for d in os.listdir(os.path.join(root_path, agent_name)) if not d.startswith('.')]
        for attempt in attempts:
            df = pd.read_csv(os.path.join(root_path, agent_name, attempt, 'history.csv'))
            df.set_index(df.columns[0], inplace=True)

            rolled = df.rolling(window=window, center=True)
            median_df = rolled.median()
            quantile25_df = rolled.quantile(0.25)
            quantile75_df = rolled.quantile(0.75)
            tmp_median_dfs.append(median_df)
            tmp_quantile25_dfs.append(quantile25_df)
            tmp_quantile75_dfs.append(quantile75_df)
        median_concat_df = pd.concat(tmp_median_dfs)
        quantile25_concat_df = pd.concat(tmp_quantile25_dfs)
        quantile75_concat_df = pd.concat(tmp_quantile75_dfs)
        median_dfs[agent_name] = median_concat_df.groupby(median_concat_df.index).mean()
        quantile25_dfs[agent_name] = quantile25_concat_df.groupby(quantile25_concat_df.index).mean()
        quantile75_dfs[agent_name] = quantile75_concat_df.groupby(quantile75_concat_df.index).mean()

    columns = median_dfs[agent_name].columns
    ncol = 2
    nrow = ceil(len(columns) / ncol)

    fig = plt.figure(figsize=(8 * ncol, 3 * nrow))
    axes = fig.subplots(nrows=nrow, ncols=ncol, sharex=True)
    for idx, col in enumerate(columns):
        ax = axes[idx // ncol][idx % ncol]
        for agent_idx, agent_name in enumerate(agent_names):
            index = median_dfs[agent_name].index
            color = f'C{agent_idx}'
            ax.plot(index, median_dfs[agent_name][col], alpha=0.8, color=color, label=agent_name)
            ax.fill_between(index, quantile25_dfs[agent_name][col], quantile75_dfs[agent_name][col], alpha=0.3, color=color)
            if idx == 0:
                ax.legend()
        ax.grid(True)
        ax.set_title(col)
        if (idx // ncol) == (nrow - 1):  # bottom ax
            ax.set_xlabel(index.name)

    ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())

    fig = ax.figure
    if title is None:
        title = ''
    fig.suptitle(title)
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    fig.savefig(out_path)
    logger.info(f'create figure at {out_path}')
    plt.close(fig)


class Recorder:
    stepwise_header = None
    episodewise_header = None
    template = None

    def __init__(self, episodewise_csv_path):
        self.episodewise_csv_path = episodewise_csv_path

    def start(self):
        self.timer = Timer()
        self._write_episodewise_header()

    def _write_episodewise_header(self):
        with open(self.episodewise_csv_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(self.episodewise_header)

    def begin_episode(self):
        self.timer.lap()
        self.episode_record = {col: [] for col in self.stepwise_header}

    def record(self, **kwargs):
        for col in self.stepwise_header:
            self.episode_record[col].append(kwargs[col])

    def end_episode(self):
        """must create a dict `self._episode_stats`"""
        raise NotImplementedError

    def dump_stepwise_csv(self, path):
        with open(path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(self.stepwise_header)
            data = zip(*(self.episode_record[col] for col in self.stepwise_header))
            writer.writerows(data)

    def dump_episodewise_csv(self):
        with open(self.episodewise_csv_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([self._episode_stats[col] for col in self.episodewise_header])

    def dump_episodewise_str(self):
        return self.template.format(**self._episode_stats)


class StandardRecorder(Recorder):
    stepwise_header = [
        'total_step', 'episode', 'episode_step', 'reward', 'action', 'action_meaning', 'is_random', 'epsilon',
        'action_q', 'loss', 'td_error']
    episodewise_header = [
        'total_step', 'episode', 'episode_step', 'reward', 'epsilon', 'action_q', 'loss', 'td_error',
        'duration', 'sps', 'total_duration']
    template = (
        'total step:{total_step:,} episode:{episode:,} epi.step:{episode_step} reward:{reward:.0f} epsilon:{epsilon:.3f} '
        'action_q:{action_q:.3f} loss:{loss:.3f} td error:{td_error:.3f} '
        'duration:{h_duration} sps:{sps:.1f} tot.dur.:{h_total_duration}')

    def end_episode(self):
        episode_step = self.episode_record['episode_step'][-1]
        assert episode_step == len(self.episode_record['episode_step'])
        duration = self.timer.lap_elapsed
        self._episode_stats = {
            'total_step': self.episode_record['total_step'][-1],
            'episode': self.episode_record['episode'][-1],
            'episode_step': episode_step,
            'reward': np.nansum(self.episode_record['reward']),
            'epsilon': self.episode_record['epsilon'][-1],
            'action_q': np.nanmean(self.episode_record['action_q']),
            'loss': np.nanmean(self.episode_record['loss']),
            'td_error': np.nanmean(self.episode_record['td_error']),
            'duration': duration,
            'h_duration': self.timer.lap_elapsed_str,
            'sps': episode_step / duration,
            'total_duration': self.timer.elapsed,
            'h_total_duration': self.timer.elapsed_str}


class Timer:
    def __init__(self):
        self._start = time.time()
        self._lap_start = self._start
        self._stop = None

    def lap(self):
        """start a new lap"""
        self._lap_start = time.time()

    def stop(self):
        self._stop = time.time()

    def tostr(self, seconds):
        sec = seconds % 60
        min_ = int((seconds // 60) % 60)
        hour = int(seconds // 3600)

        if hour > 0:
            return f'{hour}h {min_}m {sec:.0f}s'
        elif min_ > 0:
            return f'{min_}m {sec:.0f}s'
        else:
            return f'{sec:.1f}s'

    @property
    def elapsed(self):
        """total elapsed time from start (in seconds)"""
        if self._stop:
            _elapsed = self._stop - self._start
        else:
            _elapsed = time.time() - self._start
        return _elapsed

    @property
    def lap_elapsed(self):
        return time.time() - self._lap_start

    @property
    def elapsed_str(self):
        return self.tostr(self.elapsed)

    @property
    def lap_elapsed_str(self):
        return self.tostr(self.lap_elapsed)


def report_error(_logger):
    def _report_error(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                func(*args, **kwargs)
            except Exception as e:
                _logger.exception(e)
        return wrapper
    return _report_error
