import time
from functools import wraps
import traceback
import csv


class Recorder:
    def __init__(self, result_path, header, template):
        self.result_path = result_path
        self.header = header
        self.template = template

    def start(self):
        self.global_timer = Timer().start()
        self._write_header()

    def _write_header(self):
        with open(self.result_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(self.header)

    def begin_episode(self, episode):
        self.episode_timer = Timer().start()
        self.episode = episode
        self.episode_record = {col: [] for col in self.header}

    def record(self, **kwargs):
        assert kwargs.keys() == self.episode_record.keys()
        for k, v in kwargs.items():
            self.episode_record[k].append(v)

    def dump_episode_csv(self):
        with open(self.result_path, 'a') as f:
            writer = csv.writer(f)
            data = zip(*(self.data[col] for col in self.header))
            writer.writerows(data)

    def episode_summary(self):
        # duration = time.time() - self.episode_timer
        # episode_steps = self.data['episode_step'][-1]
        # assert episode_steps == len(self.data['episode_step'])
        # fps = episode_steps / duration
        # total_duration = time.time() - self.global_timer
        # episode_stats = {
        #     'step': self.data['step'][-1],
        #     'n_steps': self.n_steps,
        #     'episode': self.episode,
        #     'duration': duration,
        #     'total_duration': total_duration,
        #     'episode_steps': episode_steps,
        #     'fps': fps,
        #     'episode_reward': sum(self.data['reward']),
        #     'quest': self.data['quest'][-1],
        #     'seed': self.data['seed'][-1],
        #     'loss': sum(self.data['loss']) / episode_steps,
        #     'td_error': sum(self.data['td_error']) / episode_steps}
        # print(self.template.format(**episode_stats))
        pass


class Timer:
    def __init__(self):
        self.laptime = None

        self._start = None
        self._lap_start = None
        self._stop = None

    def start(self):
        self._start = time.time()
        self._lap_start = self._start
        return self

    def lap(self):
        lap_stop = time.time()
        self.laptime = lap_stop - self._lap_start
        self._lap_start = lap_stop

    def stop(self):
        self._stop = time.time()

    def tostr(self, seconds):
        sec = seconds % 60
        min_ = int((seconds // 60) % 60)
        hour = int(seconds // 3600)

        ret = f'{sec:.1f}sec'
        if hour > 0:
            ret = f'{hour}hour {min_}min ' + ret
        elif min_ > 0:
            ret = f'{min_}min ' + ret

        return ret

    @property
    def elapsed(self):
        """total elapsed time from start (in seconds)"""
        if self._stop:
            _elapsed = self._stop - self._start
        else:
            _elapsed = time.time() - self._start
        return _elapsed

    @property
    def elapsed_str(self):
        return self.tostr(self.elapsed)

    @property
    def laptime_str(self):
        return self.tostr(self.laptime)


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
