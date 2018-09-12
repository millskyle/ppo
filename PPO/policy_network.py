import tensorflow as tf

class PolicyNetwork(object):
    def __init__(self, env, label, temperature=0.1):
        self._sess = None
        self.o_space = env.observation_space
        self.a_space = env.action_space

        with tf.variable_scope(label):
            self.observation = tf.placeholder(dtype=tf.float32,
                                              shape=[None] + list(self.o_space.shape),
                                              name='obsservation')
            with tf.variable_scope('policy_network'):
                pi = tf.layers.dense(inputs=self.observation,
                                     units=32,
                                     activation=tf.nn.tanh)
                pi = tf.layers.dense(inputs=pi,
                                     units=32,
                                     activation=tf.nn.tanh)
                pi = tf.layers.dense(inputs=pi,
                                     units=32,
                                     activation=tf.nn.tanh)
                pi = tf.layers.dense(inputs=pi,
                                     units=self.a_space.n)
                self.a_prob = tf.layers.dense(inputs=tf.divide(pi, temperature),
                                              units=self.a_space.n,
                                              activation=tf.nn.softmax)
            with tf.variable_scope('value_net'):
                v = tf.layers.dense(inputs=self.observation,
                                    units=32,
                                    activation=tf.nn.tanh)
                v = tf.layers.dense(inputs=v,
                                    units=32,
                                    activation=tf.nn.tanh)
                self.v_preds = tf.layers.dense(inputs=v,
                                    units=1,
                                    activation=None)

            self.action_stochastic = tf.multinomial(tf.log(self.a_prob), num_samples=1)
            self.action_stochastic = tf.reshape(self.action_stochastic, shape=[-1])
            self.action_deterministic = tf.argmax(self.a_prob, axis=1)

            self.scope = tf.get_variable_scope().name

    @property
    def sess(self):
        if self._sess is None:
            logging.error("You must attach a session using the attach_session(sess) method.")
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
