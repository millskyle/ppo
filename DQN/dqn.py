import tensorflow as tf
import sys
import os
import logging
sys.path.append("..")
from supporting.NN import DenseNN
from supporting.utility import get_log_path
from supporting.utility import Buffer
from supporting.utility import Counter
from supporting.algorithm import Algorithm
import numpy as np

class Placeholders(object):
    def __init__(self, env, flags):
        state_sequence_length = flags.get('state_seq_length', 1)
        self.batch_size = tf.placeholder(tf.int64, shape=[], name='batch_size')
        self.state_in = tf.placeholder(tf.float32, shape=[None,] + [state_sequence_length,] + list(env.observation_space.shape), name='state_in')
        self.state_tp1_in = tf.placeholder(tf.float32, shape=[None,] +[state_sequence_length,] + list(env.observation_space.shape), name='next_state_in')
        self.action_in = tf.placeholder(tf.int64, shape=[None,], name='action_in')
        self.reward_in = tf.placeholder(tf.float32, shape=[None,], name='reward_in')
        self.done_in = tf.placeholder(tf.bool, shape=[None,], name='done_in')
        self.gamma = tf.placeholder(tf.float32, shape=[], name='gamma')

class DQN(Algorithm):
    def __init__(self, env, restore=True, output_path=None, flags={}):
        super().__init__(restore=restore, output_path=checkpoint_path)
        self._env = env
        self._checkpoint_path = checkpoint_path
        self._flags = flags
        self._epsilon_override = None

        self._ph = Placeholders(env=self._env, flags=self._flags)

        h = 400 #hidden layers' size
        self.online_Qnet = DenseNN(in_=self._ph.state_in,
                                   units=[h,h,self._env.action_space.n],
                                   activations=[tf.nn.selu,]*2 + [None],
                                   scope='Q',
                                   reuse=False,
                                   noisy=self._flags.get('noisy_net_magnitude', 0.0),
                                   batch_size = self._ph.batch_size,
                                   )

        if self._flags.get('double_q_learning', False):
            """Double Qnet takes the NEXT state, and uses the online
               network to predict the Q values"""
            self.double_Qnet = DenseNN(in_=self._ph.state_tp1_in,
                                       units=[h,h,self._env.action_space.n],
                                       activations=[tf.nn.selu,]*2 + [None],
                                       scope='Q',
                                       #Make sure to use the same weights!
                                       reuse=True,
                                       noisy=self._flags.get('noisy_net_magnitude', 0.0),
                                       batch_size = self._ph.batch_size,
                                       )

        self.offline_Qnet = DenseNN(in_=self._ph.state_tp1_in,
                                    units=[h,h,self._env.action_space.n],
                                    activations=[tf.nn.selu,]*2 + [None],
                                    scope='target_Q',
                                    reuse=False,
                                    noisy=self._flags.get('noisy_net_magnitude', 0.0),
                                    batch_size = self._ph.batch_size,
                                    )


        #Let's slowly build the _td_error so we can see the shapes
        #for debugging
        rj = self._ph.reward_in
        done_mask = tf.to_float(tf.logical_not(self._ph.done_in))
        max_over_actions_target_net = tf.reduce_max(self.offline_Qnet.output, axis=1)
        Q_value_targ = max_over_actions_target_net

        if self._flags.get('double_q_learning', False):
            index_of_best_action_according_to_Qnet = tf.argmax(self.double_Qnet.output, axis=1)
            q_value_of_this_action_according_to_targ_Qnet = tf.reduce_sum(tf.one_hot(index_of_best_action_according_to_Qnet, depth=self.online_Qnet.output.shape[1])*self.offline_Qnet.output, axis=1)
            Q_value_targ = q_value_of_this_action_according_to_targ_Qnet

        q_value_of_the_action_we_took = tf.reduce_sum(tf.one_hot(self._ph.action_in, depth=self.online_Qnet.output.shape[1])*self.online_Qnet.output, axis=1)

        #y_j in the DQN paper:
        yj = rj + done_mask * self._ph.gamma * Q_value_targ
        #yj = rj + self._ph.gamma * Q_value_targ

        #td-error:
        self._td_error = yj - q_value_of_the_action_we_took

        _objective = tf.reduce_mean(tf.square(self._td_error))
        tf.summary.scalar('td_error', _objective)

        self._set_up_summaries()


        logging.debug("Vars to optimize:")
        for var in self.online_Qnet.get_variables():
            logging.debug(var.name)

        logging.debug("Setting up optimizer with learning rate {}".format(self._flags.get('learning_rate', 1e-3)))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self._flags.get('learning_rate', 1e-3))
        self.train_op = self.optimizer.minimize(_objective,
                                                var_list=self.online_Qnet.get_variables(),
                                                global_step=self._weight_update_counter.var)

        self._sync_scopes_ops = self._get_sync_scopes_ops(to_scope='target_Q',
                                                          from_scope='Q')

        self._sequence_buffer = Buffer(maxlen=self._flags.get('state_seq_length',1))
        self._replay_buffer = Buffer(maxlen=self._flags.get('replay_buffer_size', 10000))
        self._episode_reward_buffer = Buffer(maxlen=None)
        self._multi_steps_buffer = Buffer(maxlen=self._flags.get('multi_steps_n', 1))

    def _get_sync_scopes_ops(self, from_scope, to_scope):
        from_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=from_scope)
        to_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=to_scope)
        assign_ops = []
        for to_var, from_var in zip(to_vars, from_vars):
            logging.debug("Creating op to sync {} --> {}".format(from_var.name, to_var.name))
            assign_ops.append(tf.assign(to_var, from_var))
        return assign_ops

    def _set_up_summaries(self):
        for var in self.online_Qnet.get_variables():
            tf.summary.histogram(var.name, var)
        for var in self.offline_Qnet.get_variables():
            tf.summary.histogram(var.name, var)
        pass



    def discount_rewards(self, gamma, R):
        """ R is a list of rewards to be discounted with R[-1]
            being the most into the future (i.e. most heavily-discounted).
            R[0] is not discounted
        """
        R_ = np.array(R) * gamma**np.arange(len(R))
        return R_

    def train(self, batch_size, epsilon, gamma, debug=False):
        data = self._replay_buffer.sample(N=batch_size, mode='prioritized')

        s, a, r, d, s_tp1  = list(map(list, zip(*data))) #transpose the list of lists
        if debug:
            print("-"*50)
            print("   in state:",s)
            print("take action:",a)
            print("    goes to:",s_tp1)
            print("-"*50)
        feed_dict = {
            self._ph.batch_size:   len(r),
            self._ph.state_in:     s,
            self._ph.action_in:    np.array(a),
            self._ph.reward_in:    np.array(r),
            self._ph.done_in:      np.array(d),
            self._ph.state_tp1_in: s_tp1,
            self._ph.gamma: gamma,
        }

        if self._weight_update_counter.eval() % 100 == 0:
            _, summaries, _td_error = self._sess.run([self.train_op, self._merged_summaries, self._td_error],
                                                     feed_dict=feed_dict)
            self._summary_writer.add_summary(summaries, self._total_step_counter.eval())
        else:
            _, _td_error = self._sess.run([self.train_op, self._td_error], feed_dict=feed_dict)


        if self._flags['prioritized_buffer']:
            self._replay_buffer.set_priorities_of_last_returned_sample(p=np.abs(_td_error))



    def get_action(self, observation, epsilon=0.0, debug=False, summary=True):


        #if self._epsilon_override is not None:
        #epsilon = min(max(self._epsilon_override, 0.0),1.0)

        if summary and self._episode_counter.eval() % 1 == 0:
            #SUMMARIES
            summary = tf.Summary()
            #summary.value.add(tag='intermediate/Q_a1', simple_value=Q_vals[0][0])
            #summary.value.add(tag='intermediate/Q_a2', simple_value=Q_vals[0][1])
            summary.value.add(tag='param/exploration', simple_value=epsilon)
            self._summary_writer.add_summary(summary, self._total_step_counter.eval())


        if epsilon > np.random.rand():
            #random action:
            a_t = self._env.action_space.sample()
        else:
            obs = np.expand_dims(observation, 0) #make the single state into a "batch" of size 1
            Q_vals = self._sess.run(self.online_Qnet.output, feed_dict={
                                   self._ph.state_in: obs,
                                   self._ph.batch_size: 1,
                                   })
            a_t = np.argmax(Q_vals)

        return a_t


#HOOKS:
    def _end_of_episode(self):
        super()._end_of_episode()

        if self._episode_counter.eval() % 1 == 0:
            #SUMMARIES
            summary = tf.Summary()
            r, _ = self._episode_reward_buffer.dump()
            summary.value.add(tag='result/reward', simple_value=sum(r))
            summary.value.add(tag='result/episode_length', simple_value=self._env_step_counter.eval())
            self._summary_writer.add_summary(summary, self._total_step_counter.eval())

        if self._episode_counter.eval() % 50 == 0:
            self._saver.save(self._sess, './' + self._checkpoint_path + '/chkpt', global_step=self._episode_counter.eval())

    def _start_of_episode(self):
        super()._start_of_episode()
        _ = self._episode_reward_buffer.empty()

    def _after_env_step(self, reward=None):
        super()._after_env_step()
        if reward is not None:
            self._episode_reward_buffer.add(reward, add_until_full=False)
