import tensorflow as tf
import sys
import os
import logging
sys.path.append("..")
from supporting.NN import DenseNN
from supporting.utility import get_log_path
from supporting.utility import Buffer
from supporting.utility import nonetoneg
import numpy as np

class Placeholders(object):
    def __init__(self, env, state_sequence_length):
        self.state_in = tf.placeholder(tf.float32, shape=[None,] + [state_sequence_length,] + list(env.observation_space.shape), name='state_in')
        self.state_tp1_in = tf.placeholder(tf.float32, shape=[None,] +[state_sequence_length,] + list(env.observation_space.shape), name='next_state_in')
        self.action_in = tf.placeholder(tf.int64, shape=[None,], name='action_in')
        self.reward_in = tf.placeholder(tf.float32, shape=[None,], name='reward_in')
        self.done_in = tf.placeholder(tf.bool, shape=[None,], name='done_in')
        self.epsilon = tf.placeholder(tf.float32, shape=[], name='current_exploration_probability')

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

    def attach_session(self, sess):
        self.__sess = sess

    def eval(self, mode='value'):
        assert self.__sess is not None, "You must attach a session to the counter by calling attach_session() before you can use the eval() method."
        return self.__sess.run(self.__mode_dict[mode])

class DQN(object):
    def __init__(self, env, restore=True, state_sequence_length=1, checkpoint_path=None, gamma=0.95):
        self._env = env
        self.__restore = restore
        self._checkpoint_path = checkpoint_path
        self._gamma = gamma

        self._episode_counter = Counter('episode')
        self._env_step_counter = Counter('env_step')
        self._total_step_counter = Counter('total_steps')
        self._weight_update_counter = Counter('weight_update')
        self.__counters=[self._episode_counter, self._env_step_counter, self._total_step_counter, self._weight_update_counter]

        self._ph = Placeholders(env=self._env, state_sequence_length=state_sequence_length)

        self.Qnet = DenseNN(in_=self._ph.state_in,
                            units=[400,400,self._env.action_space.n],
                            activations=[tf.nn.selu,]*3,# + [None],
                            scope='Q',
                            reuse=False
                            )

        self.a_t = tf.cond(pred=(self._ph.epsilon < tf.random_uniform(shape=[])),
                           true_fn=lambda: tf.argmax(self.Qnet.output)[1],
                           false_fn=lambda: tf.random_uniform(shape=[],dtype=tf.int64,minval=0, maxval=self._env.action_space.n)
                           )

        self.targ_Qnet = DenseNN(in_=self._ph.state_tp1_in,
                            units=[400,400,self._env.action_space.n],
                            activations=[tf.nn.selu,]*3,# + [None],
                            scope='target_Q',
                            reuse=False
                            )


        #Let's slowly build the _td_error so we can see the shapes
        #for debugging
        rj = self._ph.reward_in
        print(self._ph.reward_in.shape)
        done_mask = 1.0 - tf.to_float(self._ph.done_in)
        print("done_mask",done_mask.shape)
        print("target_net", self.targ_Qnet.output.shape)
        max_over_actions_target_net = tf.reduce_max(self.targ_Qnet.output, axis=1)
        print("max_over_actions_target_net", max_over_actions_target_net.shape)
        q_value_of_the_action_we_took = tf.reduce_sum(tf.one_hot(self._ph.action_in, depth=self.Qnet.output.shape[1])*self.Qnet.output, axis=1)
        print("q_value_of_the_action_we_took", q_value_of_the_action_we_took.shape)

        td_error = rj + done_mask*self._gamma * max_over_actions_target_net - q_value_of_the_action_we_took
        print("td_error",td_error.shape)

        #yj =   self._ph.reward_in + \
        #      (1.0 - tf.to_float(self._ph.done_in)) \
        #            * self._gamma \
        #            * tf.reduce_max(self.targ_Qnet.output, axis=1)
        #Q_j_a = tf.reduce_sum(tf.one_hot(self._ph.action_in, depth=self.Qnet.output.shape[1])*self.Qnet.output, axis=1)

        self._td_error = tf.reduce_mean(tf.square(td_error))

        self.objective = self._td_error
        tf.summary.scalar('td_error', self.objective)

        self._set_up_summaries()

        logging.debug("Vars to optimize:")
        for var in self.Qnet.get_variables():
            logging.debug(var.name)


        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        self.train_op = self.optimizer.minimize(self.objective,
                                                var_list=self.Qnet.get_variables(),
                                                global_step=self._weight_update_counter.var)

        self._sync_scopes_ops = self._get_sync_scopes_ops(to_scope='target_Q',
                                                          from_scope='Q')

        self._saver = tf.train.Saver()
        self._sequence_buffer = Buffer(maxlen=state_sequence_length)
        self._replay_buffer = Buffer(maxlen=1000)
        self._episode_reward_buffer = Buffer(maxlen=None)


    def _get_sync_scopes_ops(self, from_scope, to_scope):
        from_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=from_scope)
        to_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=to_scope)
        assign_ops = []
        for to_var, from_var in zip(to_vars, from_vars):
            logging.debug("Creating op to sync {} --> {}".format(from_var.name, to_var.name))
            assign_ops.append(tf.assign(to_var, from_var))
        return assign_ops

    def _set_up_summaries(self):
        for var in self.Qnet.get_variables():
            tf.summary.histogram(var.name, var)
        for var in self.targ_Qnet.get_variables():
            tf.summary.histogram(var.name, var)
        tf.summary.scalar('exploration', self._ph.epsilon)
        pass

    @property
    def sess(self):
        if self._sess is None:
            logging.error("You must attach a session using the attach_session(sess) method.")
            sys.exit(1)
        else:
            return self._sess

    def attach_session(self, sess):
        self._sess = sess
        for counter in self.__counters:
            counter.attach_session(self._sess)

        if self.__restore:
            try:
            #if True:
                logging.debug("Attempting to restore variables")
                latest_checkpoint = tf.train.latest_checkpoint(self._checkpoint_path)
                self._saver.restore(sess, latest_checkpoint)
                logging.info("Variables restored from checkpoint {}".format(latest_checkpoint))
            except:
                logging.warning("Variable restoration/ FAILED. Initializing variables instead.")
                sess.run(tf.global_variables_initializer())
            else:
                sess.run(tf.global_variables_initializer())

        self._summary_writer = tf.summary.FileWriter(get_log_path('./logs','run_'),
                                                     self._sess.graph, flush_secs=5)
        self._merged_summaries = tf.summary.merge_all()

    def train(self, batch_size, epsilon, debug=False):
        #get a batch of data randomly from the replay buffer:
        data = self._replay_buffer.sample(N=batch_size)
        s, a, r, d, s_tp1  = list(map(list, zip(*data))) #transpose the list of lists
        if debug:
            print("-"*50)
            print("   in state:",s)
            print("take action:",a)
            print("    goes to:",s_tp1)
            print("-"*50)
        feed_dict = {
            self._ph.state_in:     s,
            self._ph.action_in:    np.array(a),
            self._ph.reward_in:    np.array(r),
            self._ph.done_in:      np.array(d),
            self._ph.state_tp1_in: s_tp1,
            self._ph.epsilon:      epsilon,
        }

        if self._weight_update_counter.eval() % 100 == 0:
            _, summaries = self._sess.run([self.train_op, self._merged_summaries ], feed_dict=feed_dict)
            self._summary_writer.add_summary(summaries, self._total_step_counter.eval())
        else:
            _ = self._sess.run([self.train_op], feed_dict=feed_dict)



    def get_action(self, observation, epsilon=0.0, debug=False, summary=False):
        obs = np.expand_dims(observation, 0) #make the single state into a "batch" of size 1

        Q_vals = self._sess.run(self.Qnet.output, feed_dict={
                                   self._ph.state_in: obs
                                   })
        if summary:
            #SUMMARIES
            summary = tf.Summary()
            summary.value.add(tag='Q_a1', simple_value=Q_vals[0][0])
            summary.value.add(tag='Q_a2', simple_value=Q_vals[0][1])
            self._summary_writer.add_summary(summary, self._total_step_counter.eval())

        if epsilon > np.random.rand():
            #random action:
            a_t = self._env.action_space.sample()
        else:
            a_t = np.argmax(Q_vals)

        return a_t


#HOOKS:
    def _end_of_episode(self):
        logging.debug("End of episode {}".format(self._sess.run(self._episode_counter.val)))
        self._sess.run(self._episode_counter.inc)

        #SUMMARIES
        summary = tf.Summary()
        r, _ = self._episode_reward_buffer.dump()
        summary.value.add(tag='reward', simple_value=sum(r))
        summary.value.add(tag='episode_length', simple_value=self._env_step_counter.eval())
        self._summary_writer.add_summary(summary, self._total_step_counter.eval())

        self._saver.save(self._sess, './' + self._checkpoint_path + '/chkpt')

    def _start_of_episode(self):
        self._sess.run(self._env_step_counter.res)
        _ = self._episode_reward_buffer.empty()
        "Check to see if the 'render' file exists and set a flag"
        if os.path.exists('./render'):
            self.__render_requested = True
            os.remove('./render')
        else:
            self.__render_requested = False


    def _before_env_step(self):
        self._sess.run(self._env_step_counter.inc)
        self._sess.run(self._total_step_counter.inc)

    def _after_env_step(self, reward=None):
        if reward is not None:
            self._episode_reward_buffer.add(reward, add_until_full=False)
        if self.__render_requested:
            self._env.render()
