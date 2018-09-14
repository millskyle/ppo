import tensorflow as tf
import logging
import sys

class DenseNN(object):
    """ Creates a dense, fully-connected neural net of len(units) layers of
        width units. Output node accessible through  """
    def __init__(self, in_, units, activations, scope):
        self._in = in_
        assert len(units)==len(activations), "Each unit must have a matching activation."
        self._units = units
        self._activations = activations
        self.scope = scope

        out_ = self._in
        with tf.variable_scope(self.scope):
            for i in range(len(self._units)):
                logging.info("Building dense layer {} with {} units and {} activation.".format('layer_{0}'.format(i), self._units[i], self._activations[i]))
                out_ = tf.layers.dense(inputs=out_,
                                      units=self._units[i],
                                      activation=self._activations[i],
                                      name='layer_{0}'.format(i))
            self._output = out_
    @property
    def output(self):
        return self._output

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)




class NeuralNet(object):
    def __init__(self, env, label):
        self._sess = None


        with tf.variable_scope(label):
            self.observation = tf.placeholder(dtype=tf.float32,
                                              shape=[None] + list(env.observation_space.shape),
                                              name='observation')

            PI = DenseNN(in_=self.observation,
                         units=[32,32,32,env.action_space.n],
                         activations=[tf.nn.tanh,]*3 + [tf.nn.softmax],
                         scope='policy'

                         )
            self.a_prob = PI.output

            V = DenseNN(in_=self.observation,
                        units=[32,32,1],
                        activations=[tf.nn.tanh,]*2 + [None],
                        scope='value')
            self.v_preds = V.output

            self.action_stochastic = tf.multinomial(tf.log(self.a_prob), num_samples=1)
            self.action_stochastic = tf.reshape(self.action_stochastic, shape=[-1])
            self.action_deterministic = tf.argmax(self.a_prob, axis=1)

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
