import tensorflow as tf
import logging

class DenseNN(object):
    """ Creates a dense, fully-connected neural net of len(units) layers of
        width units. Output node accessible through  """
    def __init__(self, in_, units, activations, scope, reuse=tf.AUTO_REUSE, noisy=0.0, batch_size=None):
        self._in = in_
        assert len(units)==len(activations), "Each unit must have a matching activation."
        self._units = units
        self._activations = activations
        self.scope = scope

        out_ = tf.layers.flatten(inputs=self._in, name='flatten')
        with tf.variable_scope(self.scope, reuse=reuse):
            for i in range(len(self._units)):
                layer_name='layer_{0}'.format(i)
                logging.info("Building dense layer {} with {} units and {} activation.".format(layer_name, self._units[i], self._activations[i]))
                pre = out_
                out_ = tf.layers.dense(inputs=out_,
                                      units=self._units[i],
                                      activation=self._activations[i],
                                      name='layer_{0}'.format(i))
                if noisy>0:
                    with tf.variable_scope('noisy'):
                        out_n = tf.layers.dense(inputs=pre,
                                                units=self._units[i],
                                                activation=self._activations[i],
                                                name='noise_layer_{0}'.format(i)
                                               )
                        out_n = out_n * tf.random_uniform(shape=[batch_size]+list(out_n.shape[1:]))
                    out_ = out_ + noisy * out_n

            self._output = out_
    @property
    def output(self):
        return self._output

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
