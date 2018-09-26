import tensorflow as tf
import sys
import os
import logging
import collections
sys.path.append("..")
from supporting.NN import DenseNN
from supporting.utility import get_log_path
import numpy as np


class Buffer(object):
    def __init__(self, maxlen, prioritized=False):
        self.__maxlen=maxlen
        self.__data = collections.deque(maxlen=maxlen)
        self.__prior = collections.deque(maxlen=maxlen)

    def add(self, point, priority=1):
        self.__data.append(point)
        self.__prior.append(priority)

    def empty(self):
        D = self.__data
        P = self.__prior
        self.__data.clear()
        self.__prior.clear()
        return D, P

    def is_full(self):
        return len(self.__data)==self.__maxlen

    def pop(self, N):
        Ds = [self.__data.pop() for _ in range(N)]
        Ps = [self.__prior.pop() for _ in range(N)]
        return Ds, Ps



class Placeholders(object):
    def __init__(self, env):
        self.state_in = tf.placeholder(tf.float32, shape=[None,] + list(env.observation_space.shape), name='state_in')
        self.action_in = tf.placeholder(tf.float32, shape=env.action_space.shape, name='action_in')
        self.epsilon = tf.placeholder(tf.float32, shape=[], name='current_exploration_probability')



class Counter(object):
    def __init__(self, name, init_=0):
        self.var = tf.Variable(init_, trainable=False, name=name + '_counter') #variable
        self.val = tf.identity(self.var, name=name + '_counter_val') #get the value
        self.inc  = tf.assign(self.var, self.var + 1) #increment
        self.res = tf.assign(self.var, init_) #reset

class DQN(object):
    def __init__(self, env, restore=True, state_sequence_length=1):
        self._env = env
        self.__restore = restore

        self._episode_counter = Counter('episode')
        self._env_step_counter = Counter('env_step')


        self._ph = Placeholders(env=self._env)

        self.Qnet = DenseNN(in_=self._ph.state_in,
                            units=[64,64,self._env.action_space.n],
                            activations=[tf.nn.selu,]*2 + [None],
                            scope='Q',
                            reuse=False
                            )

        self.a_t = tf.cond(pred=(tf.random_uniform(shape=[]) > self._ph.epsilon),
                           true_fn=lambda: tf.argmax(self.Qnet.output),
                           false_fn=lambda: tf.random_uniform(shape=[],dtype=tf.int64,minval=0, maxval=self._env.action_space.n)
                           )



        self.targ_Qnet = DenseNN(in_=self._ph.state_in,
                            units=[64,64,self._env.action_space.n],
                            activations=[tf.nn.selu,]*2 + [None],
                            scope='Q_hat',
                            reuse=False
                            )

        self._sync_scopes_ops = self._get_sync_scopes_ops(to_scope='Q_hat',
                                                          from_scope='Q')

        self._saver = tf.train.Saver()

        self._sequence_buffer = Buffer(maxlen=state_sequence_length)
        self._replay_buffer = Buffer(maxlen=10000)

    def _get_sync_scopes_ops(self, from_scope, to_scope):
        from_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=from_scope)
        to_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=to_scope)
        assign_ops = []
        for to_var, from_var in zip(to_vars, from_vars):
            logging.debug("Creating op to sync {} --> {}".format(from_var.name, to_var.name))
            assign_ops.append(tf.assign(to_var, from_var))

    @property
    def sess(self):
        if self._sess is None:
            logging.error("You must attach a session using the attach_session(sess) method.")
            sys.exit(1)
        else:
            return self._sess

    def attach_session(self, sess):
        self._sess = sess

        if self.__restore:
            try:
                saver.restore(sess, tf.train.latest_checkpoint(CHKPT_PATH))
            except:
                sess.run(tf.global_variables_initializer())
            else:
                sess.run(tf.global_variables_initializer())


        self._summary_writer = tf.summary.FileWriter(get_log_path('./logs','run_'),
                                                     self._sess.graph, flush_secs=5)


    def get_action(self, observation, epsilon=0.0):
        #epsilon is probablility of random action
        a_t = self._sess.run(self.a_t,
                             feed_dict={
                                self._ph.epsilon: epsilon,
                                self._ph.state_in: observation
        })
        return 0


#HOOKS:
    def _end_of_episode(self):
        logging.debug("End of episode {}".format(self._sess.run(self._episode_counter.val)))
        self._sess.run(self._episode_counter.inc)

    def _start_of_episode(self):
        self._sess.run(self._env_step_counter.res)
        "Check to see if the 'render' file exists and set a flag"
        if os.path.exists('./render'):
            self.__render_requested = True
            os.remove('./render')
        else:
            self.__render_requested = False


    def _before_env_step(self):
        self._sess.run(self._env_step_counter.inc)

    def _after_env_step(self):
        if self.__render_requested:
            self._env.render()
