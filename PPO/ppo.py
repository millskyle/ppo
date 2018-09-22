import tensorflow as tf
import numpy as np
import logging
import sys
import copy
from utility import get_log_path
import os

class Algorithm(object):


    def make_input_placeholders(self):
        self.actions = tf.placeholder(dtype=tf.int32, shape=[None], name='actions')
        self.rewards = tf.placeholder(dtype=tf.float32, shape=[None], name='reward')
        self.v_preds_next = tf.placeholder(dtype=tf.float32, shape=[None], name='value_pred_next')
        self.advantage_estimate = tf.placeholder(dtype=tf.float32, shape=[None], name='advantage_estimate')


    def __init__(self, policy, old_policy, gamma=0.95, epsilon=0.2, c_1=1., c_2=0.01):
        """ epsilon :: clip_value """
        self.policy = policy
        self.old_policy = old_policy
        self.gamma = gamma

        self.__weight_update_counter = 0


        self._sess = None

        self.make_copy_nn_ops()  #make ops to sync old_policy to policy\
        self.make_input_placeholders() #make placeholders


        act_probs = self.policy.a_prob
        act_probs_old = self.old_policy.a_prob

        act_probs = act_probs * tf.one_hot(indices=self.actions, depth=act_probs.shape[1])
        act_probs = tf.reduce_sum(act_probs, axis=1)

        act_probs_old = act_probs_old * tf.one_hot(indices=self.actions, depth=act_probs_old.shape[1])
        act_probs_old = tf.reduce_sum(act_probs_old, axis=1)


        with tf.variable_scope('L/CLIP'):
            ratio = tf.exp(tf.log(act_probs) - tf.log(act_probs_old))
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
            L_S = -tf.reduce_mean(
                            tf.reduce_sum(
                                self.policy.a_prob*tf.log(tf.clip_by_value(self.policy.a_prob, 1e-10,1.0)),
                                axis=1
                            ),
                        axis=0
                       )

        with tf.variable_scope('Loss'):
            tf.summary.scalar('L_clip', L_clip)
            tf.summary.scalar('c_1*L_vf', c_1*L_vf)
            tf.summary.scalar('c_2*L_S', c_2*L_S)
            loss = -(L_clip - c_1*L_vf + c_2*L_S)
            #The paper says to MAXIMIZE this loss, so let's minimize the
            #negative instead

        #for var in self.old_policy.get_variables(trainable_only=True):
        #    tf.summary.histogram(var.name, var)

        for var in self.policy.get_variables(trainable_only=True):
            tf.summary.histogram(var.name, var)

        with tf.variable_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, epsilon=1e-5)
            self.train_op = optimizer.minimize(loss, var_list=self.policy.get_variables(trainable_only=True))
        self._summaries = tf.summary.merge_all()


    @property
    def sess(self):
        if self._sess is None:
            logging.error("You must attach a session using the attach_session(sess) method.")
            sys.exit(1)
        else:
            return self._sess

    def attach_session(self, sess):
        self._sess = sess
        self._summary_writer = tf.summary.FileWriter(get_log_path('./logs','run_'),
                                                     self._sess.graph, flush_secs=5)


    def train(self, observations, actions, rewards, v_preds_next, advantage_estimate):
        logging.debug("Updating weights")
        _summary, _ = self.sess.run([self._summaries, self.train_op], feed_dict={
                                                    self.policy.observation: observations,
                                                    self.old_policy.observation: observations,
                                                    self.actions: actions,
                                                    self.rewards: rewards,
                                                    self.v_preds_next: v_preds_next,
                                                    self.advantage_estimate: advantage_estimate
                                                })
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


    def estimate_advantage(self, rewards, v_preds, v_preds_next):
        deltas = [r_t + self.gamma * v_next - v for r_t, v_next, v in zip(rewards, v_preds_next, v_preds)]
        advantage_estimate = copy.deepcopy(deltas)
        for t in reversed(range(len(advantage_estimate)-1)):
            advantage_estimate[t] = advantage_estimate[t] + self.gamma * advantage_estimate[t+1]
        return advantage_estimate


    def _end_of_episode(self):
        pass


    def _start_of_episode(self):
        if os.path.exists('./render'):
            self._render = True
            os.remove('./render')
        else:
            self._render = False
