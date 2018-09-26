import tensorflow as tf
import logging

class DenseNN(object):
    """ Creates a dense, fully-connected neural net of len(units) layers of
        width units. Output node accessible through  """
    def __init__(self, in_, units, activations, scope, reuse=tf.AUTO_REUSE):
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
