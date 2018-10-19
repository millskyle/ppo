import os
import sys
import logging
import tensorflow as tf
from .utility import Counter
from .utility import get_log_path

class Algorithm(object):
    """Basic algorithm class with functionality such as hooks for before/after episodes,
       weight saving, restoring, step and episode counters, etc.
    """

    def __init__(self, restore, output_path, flags):
        self.__sess = None
        self.__restore = restore
        self._output_path = output_path
        self._checkpoint_path = self._output_path + '/chkpts'
        self._log_path = self._output_path + './logs'
        self._flags = flags

        self._episode_counter = Counter('episode')
        self._env_step_counter = Counter('env_step')
        self._total_step_counter = Counter('total_steps')
        self._weight_update_counter = Counter('weight_update')

        self.__counters = [self._episode_counter,
                           self._env_step_counter,
                           self._total_step_counter,
                           self._weight_update_counter]


        self._saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=10./60.)


    @property
    def sess(self):
        if self.__sess is None:
            logging.error("You must attach a session using the attach_session(sess) method.")
            sys.exit(1)
        else:
            return self.__sess

    @property
    def _sess(self):
        return self.sess


    def attach_session(self, sess):
        """
        1) Attaches a session to self.__sess, makes accessible via self._sess and self.session
        2) Attaches the session to counters
        3) If variables are to be restored, attempt to restore them. Otherwise,
           or upon failure initialize all variables.
        4) Set up summaries and merge summary op


        """
        self.__sess = sess
        for counter in self.__counters:
            counter.attach_session(self.__sess)

        if self.__restore:
            try:
                logging.debug("Attempting to restore variables")
                latest_checkpoint = tf.train.latest_checkpoint(self._checkpoint_path)
                self._saver.restore(sess, latest_checkpoint)
                logging.info("Variables restored from checkpoint {}".format(latest_checkpoint))
            except:
                logging.warning("Variable restoration/ FAILED. Initializing variables instead.")
                sess.run(tf.global_variables_initializer())
        else:
            sess.run(tf.global_variables_initializer())

        self._summary_writer = tf.summary.FileWriter(get_log_path(self._log_path, self._flags.get('name_prefix', 'run_')),
                                                     self._sess.graph, flush_secs=1)
        self._merged_summaries = tf.summary.merge_all()



        #Alias some of the private functions
        self.before_env_step=self._before_env_step
        self.after_env_step=self._after_env_step
        self.end_of_episode=self._end_of_episode
        self.start_of_episode=self._start_of_episode

    def reset_counters(self):
        for counter in self.__counters:
            counter.reset()


    #Hooks
    #You should call these at the appropriate times.
    def _end_of_episode(self):
        """
        1) Increment episode counter
        """
        logging.debug("End of episode {}".format(self._sess.run(self._episode_counter.val)))
        self._episode_counter.incr()


    def _start_of_episode(self):

        self._sess.run(self._env_step_counter.res)
        """Check to see if the 'render' file exists and set a flag"""
        if os.path.exists('./render'):
            self.__render_requested = True
            os.remove('./render')
        else:
            self.__render_requested = False


    def _before_env_step(self):
        self._env_step_counter.incr()
        self._total_step_counter.incr()

    def _after_env_step(self):
        if self.__render_requested:
            self._env.render()
