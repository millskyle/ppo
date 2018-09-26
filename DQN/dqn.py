import tensorflow as tf
import sys
import os
import logging
sys.path.append("..")
from supporting.NN import DenseNN
from supporting.utility import get_log_path


class Placeholders(object):
    def __init__(self, env):
        self.state_in = tf.placeholder(tf.float32, shape=[None,] + list(env.observation_space.shape), name='state_in')
        self.action_in = tf.placeholder(tf.float32, shape=env.action_space.shape, name='action_in')

class Counter(object):
    def __init__(self, name, init_=0):
        self.var = tf.Variable(init_, trainable=False, name=name + '_counter') #variable
        self.val = tf.identity(self.var, name=name + '_counter_val') #get the value
        self.inc  = tf.assign(self.var, self.var + 1) #increment
        self.res = tf.assign(self.var, init_) #reset


class DQN(object):
    def __init__(self, env, restore=True):
        self._env = env
        self.__restore = restore

        self._episode_counter = Counter('episode')
        self._env_step_counter = Counter('env_step')


        self._ph = Placeholders(env=self._env)



        self.Qnet = DenseNN(in_=self._ph.state_in,
                            units=[64,64,64],
                            activations=[tf.nn.selu,]*2 + [None],
                            scope='Qnet',
                            reuse=False
                            )

        self._saver = tf.train.Saver()

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


    def get_action(self, observation):
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
