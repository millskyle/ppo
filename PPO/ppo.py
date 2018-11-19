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
        self.actions = tf.placeholder(dtype=tf.int32, shape=[None], name='actions')
        self.rewards = tf.placeholder(dtype=tf.float32, shape=[None], name='reward')
        self.v_preds_next = tf.placeholder(dtype=tf.float32, shape=[None], name='value_pred_next')
        self.advantage_estimate = tf.placeholder(dtype=tf.float32, shape=[None], name='advantage_estimate')

    def attach_session(self, sess):
        super().attach_session(sess)
        self.policy.attach_session(sess)
        self.old_policy.attach_session(sess)


    def __init__(self, env, restore, output_path, flags, gamma=0.95, epsilon=0.2, c_1=1., c_2=0.000001,
                 use_curiosity=False, eta=0.1, llambda=0.1, beta=0.2):
        super().__init__(restore=restore, output_path=output_path, flags=flags)
        self.pins = {}
        """ epsilon :: clip_value """
        self.policy = PolicyNet(env=env, label='policy', h=8)
        self.old_policy = PolicyNet(env=env, label='old_policy', h=8)
        self.gamma = gamma
        self._use_curiosity = use_curiosity

        self.__weight_update_counter = 0

        self._buffer = Buffer(maxlen=10000, prioritized=False)

        self._env = env


        self.make_copy_nn_ops()  #make ops to sync old_policy to policy\
        self.make_input_placeholders() #make placeholders


        act_probs = self.policy.a_prob
        act_probs_old = self.old_policy.a_prob



        ### Curiosity
#        if self._use_curiosity:
#            with tf.variable_scope('curiosity'):
#                a_t = tf.cast(tf.one_hot(indices=self.actions, depth=self.policy.a_prob.shape[1], on_value=1, off_value=0), tf.float32)
#                self.action_taken_onehot = a_t
#                s_t = self.policy.observation
#                self.observation_tp1 = tf.placeholder(dtype=tf.float32,
#                                                  shape=self.policy.observation.shape,
#                                                  name='observation_tp1')
#                s_tp1 = self.observation_tp1
#
#                #encode the observation. This could be any type of neural net.
#                #if the observation is an image, probably want convolutional
#                inverse_nn_t = DenseNN(in_=s_t,
#                                     units=[16,32,64],
#                                     activations=[tf.nn.selu,]*3,
#                                     scope='curiosity_inverse')
#                #encode the observation at time t+1 with the same neural net (and
#                #same weights)
#                inverse_nn_tp1 = DenseNN(in_=s_tp1,
#                                          units=[16,32,64],
#                                          activations=[tf.nn.selu,]*3,
#                                          scope='curiosity_inverse')
#
#                joined = tf.concat((inverse_nn_t.output, inverse_nn_tp1.output), axis=1)
#
#                inverse_nn = DenseNN(in_=joined,
#                                     units=[64,a_t.shape[1]],
#                                     activations=[tf.nn.selu,None],
#                                     scope='curiosity_inverse_enc')
#                self._curiosity_a_pred = inverse_nn.output
#                inp_forward = tf.concat((self._curiosity_a_pred, s_t), axis=1)
#                forward_nn = DenseNN(in_=inp_forward,
#                                     units=[64, inverse_nn_tp1.output.shape[1]],
#                                     activations=[tf.nn.selu,] +[None],
#                                     scope='curiosity_forward'
#                                     )
#        if self._use_curiosity:
#            #we want to train the inverse network to predict the correct action:
#            with tf.variable_scope('L/Inverse'):
#                L_I = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
#                                                    logits=self._curiosity_a_pred,
#                                                    labels=a_t))
#
#                #L_I = tf.reduce_mean(tf.square(self._curiosity_a_pred-a_t))
#            with tf.variable_scope('L/Forward'):
#                L_F = tf.reduce_mean(tf.nn.l2_loss(forward_nn.output-inverse_nn_tp1.output))
#
#            self.reward_intrinsic = eta * tf.reduce_mean(tf.nn.l2_loss(forward_nn.output-inverse_nn_tp1.output))


        with tf.variable_scope('L/CLIP'):
            self.pins['act_probs'] = tf.reduce_sum(act_probs)
            self.pins['act_probs_old'] = tf.reduce_sum(act_probs_old)
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
            L_S = -tf.reduce_mean(self.policy.a_entropy )

        with tf.variable_scope('Loss'):
            tf.summary.scalar('L_clip', -L_clip*llambda)
            tf.summary.scalar('c_1*L_vf', c_1*L_vf*llambda)
            tf.summary.scalar('c_2*L_S', -c_2*L_S*llambda)
            #loss = (L_clip - c_1*L_vf + c_2*L_S)
            #The paper says to MAXIMIZE this loss, so let's minimize the
            #negative instead
            loss = -L_clip + c_1*L_vf - c_2*L_S
            self.pins["L_clip"] = L_clip
            self.pins["L_vf"] = L_vf
            self.pins["L_S"] = L_S
            if self._use_curiosity:
                loss = llambda*loss + (1-beta)*L_I + beta*L_F
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


        pin_ops = [self.pins[key] for key in sorted(self.pins)]
        pin_names = [key for key in sorted(self.pins)]

        vals = self.sess.run(pin_ops, feed_dict=feed_dict)

        formatstr = "{:>20s}"*len(vals)
        print(formatstr.format(*pin_names))

        formatstr = "{:20.5e}"*len(vals)
        print(formatstr.format(*vals))

        losses_vals = self.sess.run



        _summary, _ = self.sess.run([self._summaries, self.train_op], feed_dict=feed_dict)


        if verbose and self._use_curiosity:
            pred, taken = self.sess.run([self._curiosity_a_pred, self.action_taken_onehot], feed_dict=feed_dict)
            logging.info("A_pred:" + str(pred[0]))
            logging.info("A_taken:" + str(taken[0]))

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
