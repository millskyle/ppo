import tensorflow as tf
import numpy as np
import logging
import sys
import copy
sys.path.append("..")
from supporting.utility import get_log_path, Buffer
import os
from policy_network import DenseNN, PolicyNet
from supporting.utility import Counter
from supporting.algorithm import Algorithm

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class PPO(Algorithm):


    def make_input_placeholders(self):
        if self.policy.action_mode == "Discrete":
            self.actions = tf.placeholder(dtype=tf.int32, shape=[None], name='actions')
        else:
            self.actions = tf.placeholder(dtype=tf.float32, shape=[None], name='actions')
        self.rewards = tf.placeholder(dtype=tf.float32, shape=[None], name='reward')
        self.v_preds_next = tf.placeholder(dtype=tf.float32, shape=[None], name='value_pred_next')
        self.advantage_estimate = tf.placeholder(dtype=tf.float32, shape=[None], name='advantage_estimate')

    def attach_session(self, sess):
        super().attach_session(sess)
        self.policy.attach_session(sess)
        self.old_policy.attach_session(sess)


    def __init__(self, env, restore, output_path, flags, gamma=0.95, epsilon=0.2, c_1=1., c_2=0.000001,
                 eta=0.1, llambda=0.1, beta=0.2):
        super().__init__(restore=restore, output_path=output_path, flags=flags)
        self.scalar_pins = {}
        self.array_pins = {}
        """ epsilon :: clip_value """
        self.policy = PolicyNet(env=env, label='policy', h=32)
        self.old_policy = PolicyNet(env=env, label='old_policy', h=32)
        self.gamma = gamma

        self.__weight_update_counter = 0

        self._buffer = Buffer(maxlen=100000, prioritized=False)

        self._env = env


        self.make_copy_nn_ops()  #make ops to sync old_policy to policy\
        self.make_input_placeholders() #make placeholders




        act_probs     = self.policy.action_distribution.log_prob(self.actions)
        act_probs_old = self.old_policy.action_distribution.log_prob(self.actions)

        with tf.variable_scope('L/CLIP'):
            #self.scalar_pins['act_probs'] = tf.reduce_sum(act_probs)
            #self.scalar_pins['act_probs_old'] = tf.reduce_sum(act_probs_old)
            ratio = tf.exp(act_probs - act_probs_old)
            #ratio = tf.divide(act_probs, act_probs_old)
            ratio_clipped = tf.clip_by_value(ratio, clip_value_min=1.-epsilon, clip_value_max=1.+epsilon)
            L_clip = tf.reduce_mean(
                             tf.minimum(
                                  tf.multiply(ratio, self.advantage_estimate),
                                  tf.multiply(ratio_clipped, self.advantage_estimate)
                             )
                        )

        with tf.variable_scope('L/VF'):
            L_vf = tf.reduce_mean(tf.square(
                                self.rewards + self.gamma * self.v_preds_next
                            -    self.policy.v_preds  # V_theta(s_t)  (in paper)
                            )
                     )

        with tf.variable_scope('L/S'):
            L_S = tf.reduce_mean(self.policy.a_entropy )

        with tf.variable_scope('Loss'):
            tf.summary.scalar('L_clip', L_clip)
            tf.summary.scalar('c_1*L_vf', c_1*L_vf)
            tf.summary.scalar('c_2*L_S', c_2*L_S)
            #loss = (L_clip - c_1*L_vf + c_2*L_S)
            #The paper says to MAXIMIZE this loss, so let's minimize the
            #negative instead
            loss = -L_clip + c_1*L_vf - c_2*L_S
            self.scalar_pins["L_clip"] = L_clip
            self.scalar_pins["L_vf"] = L_vf
            self.scalar_pins["L_S"] = L_S

        vars_to_optimize = []
        for var in self.policy.get_variables(trainable_only=True):
            tf.summary.histogram(var.name, var)
            vars_to_optimize.append(var)

        with tf.variable_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, epsilon=1e-5)
            self.train_op = optimizer.minimize(loss, var_list=vars_to_optimize)
        self._summaries = tf.summary.merge_all()





    def evaluate_intrinsic_reward(self, obs, obs_tp1):
        obs_d = np.array(obs).reshape(1,-1)
        obs_tp1_d = np.array(obs).reshape(1,-1)
        r_I = self.sess.run(self.reward_intrinsic, feed_dict={self.policy.observation:obs_d,
                                                              self.observation_tp1:obs_tp1_d})
        return r_I

    def train(self, observations, actions, rewards, v_preds_next, advantage_estimate,
              observations_tp1, verbose=True):
        logging.debug("Updating weights")

        feed_dict = {
                        self.policy.observation: observations,
                        self.old_policy.observation: observations,
                        self.actions: actions,
                        self.rewards: rewards,
                        self.v_preds_next: v_preds_next,
                        self.advantage_estimate: advantage_estimate
                    }

        #run the training op, get summaries


        #for pin in self.array_pins:
    #        print(pin, self.sess.run(self.array_pins[pin], feed_dict=feed_dict))


        pin_ops = [self.scalar_pins[key] for key in sorted(self.scalar_pins)]
        pin_names = [key for key in sorted(self.scalar_pins)]

        vals = self.sess.run(pin_ops, feed_dict=feed_dict)

        formatstr = "{:>20s}"*len(vals)
        print(formatstr.format(*pin_names))

        formatstr = "{:20.5e}"*len(vals)
        print(formatstr.format(*vals))

        losses_vals = self.sess.run



        _summary, _ = self.sess.run([self._summaries, self.train_op], feed_dict=feed_dict)


        self.__weight_update_counter += 1
        if self.__weight_update_counter%10==0:
            self._summary_writer.add_summary(_summary, self.__weight_update_counter)


    def make_copy_nn_ops(self):
        with tf.variable_scope('assign_ops'):
            """Set up ops to assign the 'new' value to the 'old' variable"""
            self.assign_ops = []
            for old_variable, new_variable in zip(self.old_policy.get_variables(trainable_only=True),
                                                  self.policy.get_variables(trainable_only=True)):
                self.assign_ops.append(tf.assign(old_variable, new_variable))
            return self.assign_ops

    def assign_new_to_old(self):
        return self.sess.run(self.assign_ops)



    def truncated_general_advantage_estimate(self, T, from_buffer=False, V=None, r=None):
        """
        <int> T, truncation length
        <Buffer|False> from_buffer, whether to calculate from buffer
        <int|list(float)> V, values.  If from_buffer is a Buffer, V specifies the index of the column
           If from_buffer is  False,  V should be a list of floats of len >= T
        <int|list(float)> r, rewards.  If from_buffer is a Buffer, r specifies the index of the column
           If from_buffer is  False,  r should be a list of floats of len >= T
        """

        if from_buffer==False:
            raise NotImplementedError

        self.adv_lambda = 0.95  # TODO move this elsewhere

        Rs,_ = from_buffer.dump_column(col=r)
        Vs,_ = from_buffer.dump_column(col=V)
        Rs = np.array(Rs)
        Vs = np.array(Vs)
        Vs_tp1 = np.roll(Vs, -1)
        Vs_tp1[-1] = 0
        delta_ts = Rs + self.gamma*Vs_tp1 - Vs
        A_ts = np.array([ np.sum(delta_ts[start:start+T] * np.power(self.gamma*self.adv_lambda, np.arange((len(delta_ts[start:start+T]))))) for start in range(len(Vs))])
        return (A_ts-A_ts.mean()) / A_ts.std()
