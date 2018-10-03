import logging
import collections
import os
import numpy as np
import time
import tensorflow as tf
import progressbar



class ProgressBar(object):
    def __init__(self, maxval):
        self._maxval = maxval
        self._widgets = [ progressbar.FormatLabel('                   '), ' ',
                          progressbar.Percentage(), ' ',
                          progressbar.Bar(), ' '
                        ]
        self._bar = progressbar.ProgressBar(widgets=self._widgets, max_value=self._maxval)
        self._bar.update(0)
        self._last_val = 0

    def update(self, val):
        self._last_val = val
        self._bar.update(val)


    def status(self, status=""):
        self._widgets[0] = progressbar.FormatLabel(status)
        self._bar.update(self._last_val)

class DutyCycle(object):
    """ Return a "step function" of ones and zeros, with proportion (duty cycle)
        proportional to the probablility specified when get() is called. """
    def __init__(self, period):
        self._queue = []
        self._period = int(period)

    def get(self, prob):
        if len(self._queue)==0:
            self._queue = [np.floor((x + prob*self._period)/self._period) for x in range(self._period)]

        return self._queue.pop(0)

class Schedule(object):
    def __init__(self, name=""):
        self._name = name
        pass

    def val(self, t):
        return 0

    def plot(self, ts):
        import matplotlib.pyplot as plt
        plt.ion()
        ys = [self.val(t) for t in ts]
        plt.xlabel("Step")
        plt.ylabel(self._name)
        plt.title(self._name)
        plt.plot(ts, ys, lw=2)
        plt.pause(0.000001)




class LinearSchedule(Schedule):
    def __init__(self, start, end, steps, *args, **kwargs):
        """Linear schedule, with a y-intercept of start, decreasing to end in steps steps. If evaluated
        at t > steps, end will be returned, e.g. val = max(val, end)"""
        super().__init__(*args, **kwargs)
        self.start = start
        self.end = end
        self.steps = steps

    def val(self, t):
        v = (self.end - self.start) / float(self.steps) * t + self.start
        v = min(v, self.start)
        v = max(v, self.end)
        return v

class ExponentialSchedule(Schedule):
    def __init__(self, start, end, time_constant, base=2., *args, **kwargs):
        """Exponential schedule, starting at start, decreasing to end in steps steps. If evaluated
        at t > steps, end will be returned, e.g. val = max(val, end)"""
        super().__init__(*args, **kwargs)
        self.start = start
        self.end = end
        self.base = base
        self.time_constant = time_constant

    def val(self, t):
        v = (self.start - self.end) * self.base**(-t/self.time_constant) + self.end
        v = min(v, self.start)
        v = max(v, self.end)
        return v



def get_log_path(logdir, prefix='run_'):
    """ Given a log directory and a prefix, create a new directory,
    incrementing the ending prefix if a prefix_NN directory already exists
    in logdir """
    try:
        os.mkdir(logdir)
    except:
        pass
    dirs=[]
    for dir in os.listdir(logdir):
        if prefix in dir:
            dirs.append(dir)

    nums = [ int(s.replace(prefix, '')) for s in dirs ]
    if len(nums) > 0:
        return os.path.abspath(logdir) + '/' + prefix + str(max(nums)+1).zfill(2)
    else:
        return os.path.abspath(logdir) + '/' + prefix + str(0).zfill(2)



class ParameterOverrideFile(object):
    """Class to set up a file which gets periodically checked
       and can override a parameter (e.g. learning rate, epsilon)
    """
    def __init__(self, name, refresh_frequency=10):
        """
        Args:
          name (str) : the name of the file
          refresh_frequency (float) : the time (in seconds) between
              checking the override file.  The last known value will
              be used between checks.
        """
        self._name = name
        self._refresh_frequency = refresh_frequency
        self._tic = time.time() - 99999.  #  a long time ago

    def get(self, fallback=None):
        if (time.time() - self._tic)  > self._refresh_frequency:
            self._tic = time.time()
            try:
                logging.debug("Reading file {}".format(self._name))
                val = float(open('./' + self._name, 'r').read())
            except:
                val = fallback
            self._last_val = val
        else:
            val = self._last_val
        return val



class Counter(object):
    def __init__(self, name, init_=0):
        self.var = tf.Variable(init_, trainable=False, name=name + '_counter') #variable
        self.val = tf.identity(self.var, name=name + '_counter_val') #get the value
        self.inc  = tf.assign(self.var, self.var + 1) #increment
        self.res = tf.assign(self.var, init_) #reset
        self.__sess = None
        self.__mode_dict = {'increment':self.inc,
                            'value':self.val,
                            'reset':self.res
                            }
        self._needs_reeval = True

    def incr(self):
        self._check_session()
        self.__sess.run(self.inc)
        self._needs_reeval = True

    def attach_session(self, sess):
        self.__sess = sess

    def _check_session(self):
        assert self.__sess is not None, "You must attach a session to the counter by calling attach_session() before you can use the eval() method."

    def eval(self):
        if self._needs_reeval:
            self._check_session()
            self._last_val = self.__sess.run(self.val)
        return self._last_val









class Buffer(object):
    def __init__(self, maxlen, prioritized=False):
        self.__maxlen=maxlen
        self.__data = collections.deque(maxlen=maxlen)
        self.__prior = collections.deque(maxlen=maxlen)

    def add(self, point, priority=None, add_until_full=True):
        if priority is None:
            if self.size>0:
                priority = max(self.__prior)
            else:
                priority = 1.0
        self.__data.append(point)
        self.__prior.append(priority)
        if add_until_full:
            while not(self.is_full):
                self.__data.append(point)
                self.__prior.append(priority)

    def empty(self):
        D = list(self.__data)
        P = list(self.__prior)
        self.__data.clear()
        self.__prior.clear()
        return D, P

    @property
    def is_full(self):
        return self.size==self.__maxlen

    def pop(self, N):
        Ds = [self.__data.pop() for _ in range(N)]
        Ps = [self.__prior.pop() for _ in range(N)]
        return Ds, Ps

    def popleft(self, N):
        Ds = [self.__data.popleft() for _ in range(N)]
        Ps = [self.__prior.popleft() for _ in range(N)]
        return Ds, Ps

    def dump(self):
        """Return the data without removing from the buffer"""
        return list(self.__data), list(self.__prior)

    def set_priorities_of_last_returned_sample(self, p):
        for p_index, buffer_index in enumerate(self._last_returned_indices):
            self.__prior[buffer_index] = p[p_index]

    @property
    def size(self):
        return len(self.__data)

    def sample(self, N, mode='random'):
        if mode=='random':
            indices = np.random.choice(len(self.__data), size=N)
            batch = [self.__data[i] for i in indices]
            return batch
        elif mode=='prioritized':
            p_normed = np.array(self.__prior)
            p_normed = p_normed / p_normed.sum()
            indices = np.random.choice(len(self.__data), size=N, p=p_normed)
            batch = [self.__data[i] for i in indices]
        else:
            logging.error("Sampling mode {} unrecognized".format(mode))
            raise NotImplementedError
        self._last_returned_indices = indices
        return batch
