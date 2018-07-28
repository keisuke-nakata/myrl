import time
from functools import wraps
import traceback


class Timer:
    def __init__(self):
        self.laptime = None

        self._start = None
        self._lap_start = None
        self._stop = None

    def start(self):
        self._start = time.time()
        self._lap_start = self._start

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


def report_error(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            f(*args, **kwargs)
        except Exception as e:
            traceback.print!!!!
