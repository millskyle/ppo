import collections
import os
import numpy as np


class LinearSchedule(object):
    def __init__(self, start, end, steps):
        self.start = start
        self.end = end
        self.steps = steps

    def val(self, t):
        return (self.end - self.start) / float(self.steps) * t + self.start





def get_log_path(logdir, prefix='run_'):
    try:
        os.mkdir(logdir)
    except:
        pass
    nums = [ int(s.replace(prefix, '')) for s in os.listdir(logdir) ]
    if len(nums) > 0:
        return os.path.abspath(logdir) + '/' + prefix + str(max(nums)+1).zfill(2)
    else:
        return os.path.abspath(logdir) + '/' + prefix + str(0).zfill(2)

class Buffer(object):
    def __init__(self, maxlen, prioritized=False):
        self.__maxlen=maxlen
        self.__data = collections.deque(maxlen=maxlen)
        self.__prior = collections.deque(maxlen=maxlen)

    def add(self, point, priority=1, add_until_full=True):
        self.__data.append(point)
        self.__prior.append(priority)
        if add_until_full:
            while not(self.is_full()):
                self.__data.append(point)
                self.__prior.append(priority)


    def empty(self):
        D = list(self.__data)
        P = list(self.__prior)
        self.__data.clear()
        self.__prior.clear()
        return D, P

    def is_full(self):
        return len(self.__data)==self.__maxlen

    def pop(self, N):
        Ds = [self.__data.pop() for _ in range(N)]
        Ps = [self.__prior.pop() for _ in range(N)]
        return Ds, Ps

    def dump(self):
        """Return the data without removing from the buffer"""
        return list(self.__data), list(self.__prior)

    @property
    def size(self):
        return len(self.__data)

    def sample(self, N, random=True):
        if random:
            indices = np.random.choice(len(self.__data), size=N)
            batch = [self.__data[i] for i in indices]
            return batch
