import tensorflow as tf
import gym
import logging
import numpy as np
import sys
sys.path.append("..")
from supporting.NN import DenseNN

class PolicyNet(object):
    def __init__(self, env, label, h=64):
        #h: hidden unit size
        self._sess = None

        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        if isinstance(env.action_space, gym.spaces.Discrete):
            print("!!!  DISCRETE ACTION SPACE  !!!")
            self.action_mode = "Discrete"
        elif isinstance(env.action_space, gym.spaces.Box):
            print("!!! CONTINUOUS ACTION SPACE !!!")
            self.action_mode = "Continuous"
        else:
            print(f"Algorithm not implemented for action space type {env.action_space}")
            raise NotImplementedError
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        with tf.variable_scope(label):
            self.observation = tf.placeholder(dtype=tf.float32,
                                              shape=[None] + list(env.observation_space.shape),
                                              name='observation')

            if self.action_mode == 'Discrete':
                policy_out_size = env.action_space.n
            elif self.action_mode == 'Continuous':
                policy_out_size = np.prod(env.action_space.shape)

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
                logsigma = tf.get_variable("action_sigma",
                            shape=(policy_out_size,),
                            dtype=tf.float32,
                            initializer=tf.zeros_initializer, trainable=True)
                #sigma = tf.nn.sigmoid(sigma) #make sure it never goes/starts negative
                mu = PI.output
                self.action_distribution = tf.distributions.Normal(loc=mu,
                                                                   scale=tf.exp(logsigma*-2),
                                                                   allow_nan_stats=False)
                self.action_deterministic = mu
                self.action_stochastic = self.action_distribution.sample()

                self.action_distribution_sigma = logsigma
                self.action_distribution_mean = mu

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
        logging.debug(f"observation.shape={observation.shape} (data)")
        logging.debug(f"self.observation.shape={self.observation.shape} (placeholder)")
        return self.sess.run([action_op, self.v_preds], feed_dict={self.observation: observation})

    def get_action_probabilities(self, observation):
        self.sess.run(self.a_prob, feed_dict={self.observation: observation})

    def get_variables(self, trainable_only=False):
        if trainable_only:
            gk = tf.GraphKeys.TRAINABLE_VARIABLES
        else:
            gk = tf.GraphKeys.GLOBAL_VARIABLES
        return tf.get_collection(gk, self.scope)
