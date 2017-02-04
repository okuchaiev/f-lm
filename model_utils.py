import math
import tensorflow as tf
import random
from tensorflow import tanh, sigmoid, identity
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops

def variable_summaries(var, groupname, name):
    """Attach a lot of summaries to a Tensor.
        This is also quite expensive.
    """
    with tf.device("/cpu:0"), tf.name_scope(None):
        s_var = tf.cast(var, tf.float32)
        amean = tf.reduce_mean(tf.abs(s_var))
        tf.summary.scalar(groupname + '/amean/' + name, amean)
        mean = tf.reduce_mean(s_var)
        tf.summary.scalar(groupname + '/mean/' + name, mean)
        stddev = tf.sqrt(tf.reduce_sum(tf.square(s_var - mean)))
        tf.summary.scalar(groupname + '/sttdev/' + name, stddev)
        tf.summary.scalar(groupname + '/max/' + name, tf.reduce_max(s_var))
        tf.summary.scalar(groupname + '/min/' + name, tf.reduce_min(s_var))
        tf.summary.histogram(groupname + "/" + name, s_var)

def getdtype(hps, is_rnn=False):
    if is_rnn:
        return tf.float16 if hps.float16_rnn else tf.float32
    else:
        return tf.float16 if hps.float16_non_rnn else tf.float32



def sharded_variable(name, shape, num_shards, dtype=tf.float32, transposed=False):
    # The final size of the sharded variable may be larger than requested.
    # This should be fine for embeddings.
    shard_size = int((shape[0] + num_shards - 1) / num_shards)
    if transposed:
        initializer = tf.uniform_unit_scaling_initializer(dtype=dtype)
    else:        
        initializer = tf.uniform_unit_scaling_initializer(dtype=dtype)
    return [tf.get_variable(name + "_" + str(i), [shard_size, shape[1]],
                            initializer=initializer, dtype=dtype) for i in range(num_shards)]


# XXX(rafal): Code below copied from rnn_cell.py
def _get_sharded_variable(name, shape, dtype, num_shards):
    """Get a list of sharded variables with the given dtype."""
    if num_shards > shape[0]:
        raise ValueError("Too many shards: shape=%s, num_shards=%d" %
                         (shape, num_shards))
    unit_shard_size = int(math.floor(shape[0] / num_shards))
    remaining_rows = shape[0] - unit_shard_size * num_shards

    shards = []
    for i in range(num_shards):
        current_size = unit_shard_size
        if i < remaining_rows:
            current_size += 1
        shards.append(tf.get_variable(name + "_%d" % i, [current_size] + shape[1:], dtype=dtype))
    return shards


def _get_concat_variable(name, shape, dtype, num_shards):
    """Get a sharded variable concatenated into one tensor."""
    _sharded_variable = _get_sharded_variable(name, shape, dtype, num_shards)
    if len(_sharded_variable) == 1:
        return _sharded_variable[0]

    return tf.concat(_sharded_variable, 0)

#taken from tensorflow/contrib/rnn/python/ops/core_rnn_cell_impl.py
def _linear(args, output_size, bias, bias_start=0.0):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape() for a in args]
  for shape in shapes:
    if shape.ndims != 2:
      raise ValueError("linear is expecting 2D arguments: %s" % shapes)
    if shape[1].value is None:
      raise ValueError("linear expects shape[1] to be provided for shape %s, "
                       "but saw %s" % (shape, shape[1]))
    else:
      total_arg_size += shape[1].value

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  scope = vs.get_variable_scope()
  with vs.variable_scope(scope) as outer_scope:
    weights = vs.get_variable(
        "weights", [total_arg_size, output_size], dtype=dtype)
    if len(args) == 1:
      res = math_ops.matmul(args[0], weights)
    else:
      res = math_ops.matmul(array_ops.concat(args, 1), weights)
    if not bias:
      return res
    with vs.variable_scope(outer_scope) as inner_scope:
      inner_scope.set_partitioner(None)
      biases = vs.get_variable(
          "biases", [output_size],
          dtype=dtype,
          initializer=init_ops.constant_initializer(bias_start, dtype=dtype))
    return nn_ops.bias_add(res, biases)



class GLSTMCell(tf.contrib.rnn.RNNCell):
    """LSTM cell with groups"""
    def __init__(self, num_units, initializer=None,
                 num_proj=None, number_of_groups=1, forget_bias=1.0,
                 activation=tanh):

        self._num_units = num_units
        self._initializer = initializer
        self._num_proj = num_proj
        self._forget_bias = forget_bias
        self._number_of_groups = number_of_groups
        self._activation = activation

        assert(self._num_units % self._number_of_groups == 0)
        assert(self._num_proj % self._number_of_groups == 0)
        self._group_shape = [self._num_proj / self._number_of_groups,
                             self._num_units / self._number_of_groups]
        print('GLSTM cell group shape: ' + str(self._group_shape))

        if num_proj:
            self._state_size = (tf.contrib.rnn.LSTMStateTuple(num_units, num_proj))
            self._output_size = num_proj
        else:
            self._state_size = (tf.contrib.rnn.LSTMStateTuple(num_units, num_units))
            self._output_size = num_units

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def _get_input_for_group(self, inpt, group_id, group_size):
        return tf.slice(inpt, [0, group_id*group_size], [inpt.get_shape()[0].value, group_size])

    def __call__(self, inputs, state, scope=None):

        dtype = inputs.dtype #infter GLSTM type from type of its inputs
        input_size = inputs.get_shape().with_rank(2)[1]
        if input_size.value is None:
            raise ValueError("Could not infer input size from inputs.get_shape()[-1]")
        (c_prev, m_prev) = state
        with vs.variable_scope(scope or "GLSTM_cell", initializer=self._initializer):
            i_parts = []
            j_parts = []
            f_parts = []
            o_parts = []
        
            inputs_g = tf.split(value=inputs, num_or_size_splits=self._number_of_groups, axis=1, name="splitInput")
            m_prev_g = tf.split(value=m_prev, num_or_size_splits=self._number_of_groups, axis=1, name="splitMprev")

            for group_id in xrange(self._number_of_groups):
                with vs.variable_scope("G_%d" % group_id) as g_scope:                
                    R_k = _linear([inputs_g[group_id], m_prev_g[group_id]], 4*self._group_shape[1], bias=False)
                    i_k, j_k, f_k, o_k = tf.split(R_k, 4, 1)
                    i_parts.append(i_k)
                    j_parts.append(j_k)
                    f_parts.append(f_k)
                    o_parts.append(o_k)
            
            #biases for gates
            self._b_i = tf.get_variable(
                "B_i", shape=[self._num_units], dtype=dtype)
            self._b_j = tf.get_variable(
                "B_j", shape=[self._num_units], dtype=dtype)
            self._b_f = tf.get_variable(
                "B_f", shape=[self._num_units], dtype=dtype)
            self._b_o = tf.get_variable(
                "B_o", shape=[self._num_units], dtype=dtype)

            i = tf.concat(i_parts, axis=1) + self._b_i
            j = tf.concat(j_parts, axis=1) + self._b_j
            f = tf.concat(f_parts, axis=1) + self._b_f
            o = tf.concat(o_parts, axis=1) + self._b_o

            c = tf.sigmoid(f + self._forget_bias) * c_prev + tf.sigmoid(i) * self._activation(j)
            m = tf.sigmoid(o) * self._activation(c)
            
            if self._num_proj is not None:
                with vs.variable_scope("projection") as proj_scope:
                    m = _linear(m, self._num_proj, bias=False)

            new_state = (tf.contrib.rnn.LSTMStateTuple(c, m))
            return m, new_state


class DLSTMCell(tf.contrib.rnn.RNNCell):
    """LSTM cell with groups with FC DNN inside"""
    def __init__(self, num_units, initializer=None,
                 num_proj=None, forget_bias=1.0,
                 activation=tanh,
                 hlayers=None, layer_activation=tf.nn.relu,
                 do_layer_norm=False):

        self._num_units = num_units
        self._initializer = initializer
        self._num_proj = num_proj
        self._forget_bias = forget_bias
        self._activation = activation
        self._hlayers = hlayers
        self._layer_activation = layer_activation
        self._do_layer_norm = do_layer_norm

        if num_proj:
            self._state_size = (tf.contrib.rnn.LSTMStateTuple(num_units, num_proj))
            self._output_size = num_proj
        else:
            self._state_size = (tf.contrib.rnn.LSTMStateTuple(num_units, num_units))
            self._output_size = num_units
        
        assert(self._hlayers is not None)
        print('DLSTM cell topology: ' + str(self._hlayers))
        print('DLSTM cell interlayer activations: ' + str(self._layer_activation))

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def _get_input_for_group(self, inpt, group_id, group_size):
        return tf.slice(inpt, [0, group_id*group_size], [inpt.get_shape()[0].value, group_size])

    def __call__(self, inputs, state, scope=None):

        dtype = inputs.dtype #infter GLSTM type from type of its inputs
        input_size = inputs.get_shape().with_rank(2)[1]
        if input_size.value is None:
            raise ValueError("Could not infer input size from inputs.get_shape()[-1]")
        (c_prev, m_prev) = state
        finput = tf.concat([inputs, m_prev], axis=1)
        with vs.variable_scope(scope or "DLSTM_cell", initializer=self._initializer):            
            for l in xrange(len(self._hlayers)):
                layer_size = int(self._hlayers[l])
                with vs.variable_scope("fc_layer_%d" % l) as layer_scope:
                    if self._do_layer_norm:
                        finput = tf.contrib.layers.fully_connected(inputs=finput, num_outputs=layer_size, normalizer_fn=tf.contrib.layers.layer_norm,
                                                                   activation_fn=self._layer_activation,
                                                                   reuse=True, scope=layer_scope)
                    else:
                        finput = tf.contrib.layers.fully_connected(inputs=finput, num_outputs=layer_size,
                                                                   activation_fn=self._layer_activation,
                                                                   reuse=True, scope=layer_scope)
            #output layer
            with vs.variable_scope("fc_last_layer") as layer_scope:
                lstm_matrix = tf.contrib.layers.fully_connected(inputs=finput, num_outputs=self._num_units*4,
                                                                activation_fn=identity,
                                                                reuse=True, scope=layer_scope)
            
            i, j, f, o = tf.split(value=lstm_matrix, num_or_size_splits=4, axis=1)                                                                   
            c = tf.sigmoid(f + self._forget_bias) * c_prev + tf.sigmoid(i) * self._activation(j)
            m = tf.sigmoid(o) * self._activation(c)
            
            if self._num_proj is not None:
                with vs.variable_scope("projection") as proj_scope:
                    m = _linear(m, self._num_proj, bias=False)

            new_state = (tf.contrib.rnn.LSTMStateTuple(c, m))
            return m, new_state


