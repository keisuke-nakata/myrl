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


def visualize(csv_path, out_path=None, window=10):
    if out_path is None:
        out_path = os.path.splitext(csv_path)[0] + '.png'
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    df.set_index(df.columns[0], inplace=True)
    smoothed_df = df.rolling(window=window, center=True).mean()

    ncol = 2
    nrow = ceil(len(smoothed_df.columns) / ncol)
    axes = smoothed_df.plot(subplots=True, sharex=True, grid=True, figsize=(8 * ncol, 3 * nrow), layout=(nrow, ncol))

    axes.item(0).xaxis.set_major_formatter(mpl.ticker.EngFormatter())

    fig = axes.item(0).figure
    fig.tight_layout()
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
    stepwise_header = ['total_step', 'episode', 'episode_step', 'reward', 'action', 'is_random', 'epsilon', 'loss', 'td_error']
    episodewise_header = ['total_step', 'episode', 'episode_step', 'reward', 'epsilon', 'loss', 'td_error', 'duration', 'sps', 'total_duration']
    template = (
        'total step:{total_step:,} episode:{episode} epi.step:{episode_step} reward:{reward:.0f} epsilon:{epsilon:.3f} '
        'loss:{loss:.5f} td error:{td_error:.5f} '
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
