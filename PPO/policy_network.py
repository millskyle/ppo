import tensorflow as tf
import gym
import logging
import numpy as np
import sys
sys.path.append("..")
from supporting.NN import DenseNN

#class DenseNN(object):
#    """ Creates a dense, fully-connected neural net of len(units) layers of
#        width units. Output node accessible through  """
#    def __init__(self, in_, units, activations, scope, reuse=tf.AUTO_REUSE):
#        self._in = in_
#        assert len(units)==len(activations), "Each unit must have a matching activation."
#        self._units = units
#        self._activations = activations
#        self.scope = scope
#
#        out_ = self._in
#        with tf.variable_scope(self.scope, reuse=reuse):
#            for i in range(len(self._units)):
#                layer_name='layer_{0}'.format(i)
#                logging.info("Building dense layer {} with {} units and {} activation.".format(layer_name, self._units[i], self._activations[i]))
#                out_ = tf.layers.dense(inputs=out_,
#                                      units=self._units[i],
#                                      activation=self._activations[i],
#                                      name='layer_{0}'.format(i))
#            self._output = out_
#    @property
#    def output(self):
#        return self._output
#
#    def get_variables(self):
#        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)




class PolicyNet(object):
    def __init__(self, env, label, h=64):
        #h: hidden unit size
        self._sess = None

        if isinstance(env.action_space, gym.spaces.Discrete):
            self.action_mode = "Discrete"
        elif isinstance(env.action_space, gym.spaces.Box):
            self.action_mode = "Continuous"
        else:
            print(f"Algorithm not implemented for action space type {env.action_space}")
            raise NotImplementedError

        with tf.variable_scope(label):
            self.observation = tf.placeholder(dtype=tf.float32,
                                              shape=[None] + list(env.observation_space.shape),
                                              name='observation')

            if self.action_mode == 'Discrete':
                policy_out_size = env.action_space.n
            elif self.action_mode == 'Continuous':
                policy_out_size = np.prod(env.action_space.shape) * 2  #*2 for mu and sigma

            PI = DenseNN(in_=self.observation,
                         units=[h,h,policy_out_size],
                         activations=[tf.nn.tanh,]*2 + [None],
                         scope='policy'
                         )

            if self.action_mode=='Discrete':

                self.action_distribution = tf.distributions.Categorical(probs=tf.nn.softmax(PI.output), validate_args=True)
                self.action_stochastic = self.action_distribution.sample()
                self.action_deterministic = tf.argmax(PI.output, axis=1)

            elif self.action_mode == 'Continuous':
                #TODO: Implement continuous action space here, e.g. take the output
                #of PI as Mu and Sigma of a distribution and sample from that,
                #example pseudocodei
                mu, sigma = tf.split(value=PI.output, num_or_size_splits=2, axis=1)
                sigma = tf.softmax(sigma)
                self.action_distribution = tf.distributions.Normal(mu,sigma, allow_nan_stats=False)
                self.action_deterministic = mu
                self.action_stochastic = self.action_distribution.sample()

            self.a_entropy = self.action_distribution.entropy()


            #Value net
            V = DenseNN(in_=self.observation,
                        units=[h,h,1],
                        activations=[tf.nn.tanh,]*2 + [None],
                        scope='value')
            self.v_preds = V.output
            self.scope = tf.get_variable_scope().name

    @property
    def sess(self):
        if self._sess is None:
            logging.error("You must attach a session using the attach_session(sess) method.")
            sys.exit(1)
        else:
            return self._sess

    def attach_session(self,sess):
        self._sess = sess

    def act(self, observation, stochastic):
        if stochastic:
            action_op = self.action_stochastic
        else:
            action_op = self.action_deterministic

        return self.sess.run([action_op, self.v_preds], feed_dict={self.observation: observation})

    def get_action_probabilities(self, observation):
        self.sess.run(self.a_prob, feed_dict={self.observation: observation})

    def get_variables(self, trainable_only=False):
        if trainable_only:
            gk = tf.GraphKeys.TRAINABLE_VARIABLES
        else:
            gk = tf.GraphKeys.GLOBAL_VARIABLES
        return tf.get_collection(gk, self.scope)
