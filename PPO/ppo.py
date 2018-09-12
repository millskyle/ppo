import tensorflow as tf
import numpy as np
import logging





class PPO(object):
    def __init__(self, policy, old_policy, gamma=0.95, epsilon=0.2, c_1=1, c_2=0.01):
        """ epsilon :: clip_value """
        self.policy = policy
        self.old_policy = old_policy
        self.gamma = gamma


        self._sess = None


        with tf.variable_scope('assign_ops'):
            """Set up ops to assign the 'new' value to the 'old' variable"""
            self.assign_ops = []
            for old_variable, new_variable in zip(self.old_policy.get_variables(trainable_only=True),
                                                  self.policy.get_variables(trainable_only=True)):
                self.assign_ops.append(tf.assign(old_variable, new_variable))

        with tf.variable_scope('training_input'):
            self.actions = tf.placeholder(dtype=tf.int32, shape=[None], name='actions')
            self.rewards = tf.placeholder(dtype=tf.float32, shape=[None], name='reward')
            self.v_preds_next = tf.placeholder(dtype=tf.float32, shape=[None], name='value_pred_next')
            self.advantage_estimate = tf.placeholder(dtype=tf.float32, shape=[None], name='advantage_estimate')

        act_probs = self.policy.a_prob
        act_probs_old = self.policy.a_prob

        act_probs = act_probs * tf.one_hot(indices=self.actions, depth=act_probs.shape[1])
        act_probs = tf.reduce_sum(act_probs, axis=1)

        act_probs_old = act_probs_old * tf.one_hot(indices=self.actions, depth=act_probs_old.shape[1])
        act_probs_old = tf.reduce_sum(act_probs_old, axis=1)


        with tf.variable_scope('L/CLIP'):
            ratio = tf.divide(act_probs, act_probs_old)
            ratio_clipped = tf.clip_by_value(ratio, clip_value_min=1.-epsilon, clip_value_max=1.+epsilon)
            L_clip = tf.reduce_mean(
                             tf.minimum(
                                  tf.multiply(ratio, self.advantage_estimate),
                                  tf.multiply(ratio_clipped, tf.advantage_estimate)
                             )
                        )



        with tf.variable_scope('L/VF'):
            L_vf = tf.reduce_mean(
                            tf.squared_difference(
                                self.rewards + self.gamma * self.v_preds_next,
                                self.policy.v_preds
                            )
                     )

        with tf.variable_scope('L/S'):
            L_S = -tf.reduce_mean(
                            tf.reduce_sum(
                                self.policy.a_prob*tf.log(tf.clip_by_value(self.policy.act_probs, 1e-10,1.0)),
                                axis=1
                            ),
                        axis=0
                       )

        with tf.variable_scope('L'):
            loss = L_clip - c_1*L_vf + c_2*L_S
            #The paper says to MAXIMIZE this loss, so let's minimize the
            #negative instead
            loss = -loss

        with tf.variable_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, epsilon=1e-5)
            train_op = optimizer.minimize(loss, var_list=self.policy.get_variables(trainable_only=True))

    @property
    def sess(self):
        if self._sess is None:
            logging.error("You must attach a session using the attach_session(sess) method.")
        else:
            return self._sess

    def attach_session(self, sess):
        self._sess = session


    def train(self, observations, actions, rewards, v_preds_next, advantage_estimate):
        self.sess.run(self.train_op, feed_dict={
                                                    self.policy.observation: observation,
                                                    self.old_policy.observation: observation,
                                                    self.actions: actions,
                                                    self.rewards: rewards,
                                                    self.v_preds_next: v_preds_next,
                                                    self.advantage_estimate: advantage_estimate
                                                })
    def assign_new_to_old(self):
        return self.sess.run(self.assign_ops)


    def estimate_advantage(self, rewards, v_preds, v_preds_next):
        deltas = [r_t + self.gamma * v_next - v for r_t, v_next, v in zip(rewards, v_preds_next, v_preds)]
        advantage_estimate = copy.deepcopy(deltas)
        for t in reversed(range(len(advantage_estimate)-1)):
            advantage_estimate[t] = advantage_estimate[t] + self.gamma * advantage_estimate[t+1]
        return advantage_estimate
