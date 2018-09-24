import tensorflow as tf
import numpy as np
import logging
import sys
import copy
from utility import get_log_path
import os
from policy_network import DenseNN

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class Algorithm(object):


    def make_input_placeholders(self):
        self.actions = tf.placeholder(dtype=tf.int32, shape=[None], name='actions')
        self.rewards = tf.placeholder(dtype=tf.float32, shape=[None], name='reward')
        self.v_preds_next = tf.placeholder(dtype=tf.float32, shape=[None], name='value_pred_next')
        self.advantage_estimate = tf.placeholder(dtype=tf.float32, shape=[None], name='advantage_estimate')


    def __init__(self, policy, old_policy, gamma=0.95, epsilon=0.2, c_1=1., c_2=0.01,
                 use_curiosity=False, eta=0.1, llambda=0.1, beta=0.2):
        """ epsilon :: clip_value """
        self.policy = policy
        self.old_policy = old_policy
        self.gamma = gamma
        self._use_curiosity = use_curiosity

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




        ### Curiosity
        if self._use_curiosity:
            with tf.variable_scope('curiosity'):
                a_t = self.policy.a_prob
                s_t = self.policy.observation
                self.observation_tp1 = tf.placeholder(dtype=tf.float32,
                                                  shape=self.policy.observation.shape,
                                                  name='observation_tp1')
                s_tp1 = self.observation_tp1

                #encode the observation. This could be any type of neural net.
                #if the observation is an image, probably want convolutional
                inverse_nn_t = DenseNN(in_=s_t,
                                     units=[16,16,32],
                                     activations=[tf.nn.selu,]*3,
                                     scope='curiosity_inverse')
                #encode the observation at time t+1 with the same neural net (and
                #same weights)
                inverse_nn_tp1 = DenseNN(in_=s_tp1,
                                          units=[16,16,32],
                                          activations=[tf.nn.selu,]*3,
                                          scope='curiosity_inverse')

                joined = tf.concat((inverse_nn_t.output, inverse_nn_tp1.output), axis=1)

                inverse_nn = DenseNN(in_=joined,
                                     units=[32,a_t.shape[1]],
                                     activations=[tf.nn.selu,None],
                                     scope='curiosity_inverse_enc')
                self._curiosity_a_pred = inverse_nn.output
                inp_forward = tf.concat((self._curiosity_a_pred, s_t), axis=1)
                forward_nn = DenseNN(in_=inp_forward,
                                     units=[64,64, inverse_nn_tp1.output.shape[1]],
                                     activations=[tf.nn.selu,]*2 +[None],
                                     scope='curiosity_forward'
                                     )
        if self._use_curiosity:
            #we want to train the inverse network to predict the correct action:
            with tf.variable_scope('L/Inverse'):
                L_I = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self._curiosity_a_pred,
                                                             labels=a_t))
            with tf.variable_scope('L/Forward'):
                L_F = tf.reduce_mean(tf.nn.l2_loss(forward_nn.output-inverse_nn_tp1.output))

            self.reward_intrinsic = eta * L_F


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
            tf.summary.scalar('L_clip', -L_clip*llambda)
            tf.summary.scalar('c_1*L_vf', c_1*L_vf*llambda)
            tf.summary.scalar('c_2*L_S', -c_2*L_S*llambda)
            #loss = (L_clip - c_1*L_vf + c_2*L_S)
            #The paper says to MAXIMIZE this loss, so let's minimize the
            #negative instead
            loss = -L_clip + c_1*L_vf - c_2*L_S
            if self._use_curiosity:
                loss = loss + (1-beta)*L_I + beta*L_F
                tf.summary.scalar('L_inverse', (1-beta)*L_I )
                tf.summary.scalar('L_forward', beta*L_F)

        vars_to_optimize = []
        for var in self.policy.get_variables(trainable_only=True):
            tf.summary.histogram(var.name, var)
            vars_to_optimize.append(var)

        if self._use_curiosity:
            for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='curiosity'):
                tf.summary.histogram(var.name, var)
                vars_to_optimize.append(var)

        with tf.variable_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, epsilon=1e-5)
            self.train_op = optimizer.minimize(loss, var_list=vars_to_optimize)
            #icm_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, epsilon=1e-5)
            #self.icm_train_op = icm_optimizer.minimize()
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
        if self._use_curiosity:
            feed_dict[self.observation_tp1] = observations_tp1

        #run the training op, get summaries
        _summary, _ = self.sess.run([self._summaries, self.train_op], feed_dict=feed_dict)


        if verbose:
            pred, taken = self.sess.run([self._curiosity_a_pred, self.policy.a_prob], feed_dict=feed_dict)
            logging.info("A_pred:" + str(softmax(pred[0])))
            logging.info("A_taken:" + str(softmax(taken[0])))

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
