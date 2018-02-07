"""Module for constructing RNN Cells."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

from tensorflow.contrib.compiler import jit
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest

# pylint: disable=protected-access
_Linear = core_rnn_cell._Linear  # pylint: disable=invalid-name
# pylint: enable=protected-access

class FLSTMCell(rnn_cell_impl.RNNCell):
  """Group LSTM cell (G-LSTM).
  The implementation is based on:
    https://arxiv.org/abs/1703.10722
  O. Kuchaiev and B. Ginsburg
  "Factorization Tricks for LSTM Networks", ICLR 2017 workshop.
  """

  def __init__(self, num_units, fact_size, initializer=None, num_proj=None,
               forget_bias=1.0, activation=math_ops.tanh,
               reuse=None):
    """Initialize the parameters of G-LSTM cell.
    Args:
      num_units: int, The number of units in the G-LSTM cell
      initializer: (optional) The initializer to use for the weight and
        projection matrices.
      num_proj: (optional) int, The output dimensionality for the projection
        matrices.  If None, no projection is performed.
      number_of_groups: (optional) int, number of groups to use.
        If `number_of_groups` is 1, then it should be equivalent to LSTM cell
      forget_bias: Biases of the forget gate are initialized by default to 1
        in order to reduce the scale of forgetting at the beginning of
        the training.
      activation: Activation function of the inner states.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already
        has the given variables, an error is raised.
    Raises:
      ValueError: If `num_units` or `num_proj` is not divisible by
        `number_of_groups`.
    """
    super(FLSTMCell, self).__init__(_reuse=reuse)
    self._num_units = num_units
    self._initializer = initializer
    self._fact_size = fact_size
    self._forget_bias = forget_bias
    self._activation = activation
    self._num_proj = num_proj

    if num_proj:
      self._state_size = rnn_cell_impl.LSTMStateTuple(num_units, num_proj)
      self._output_size = num_proj
    else:
      self._state_size = rnn_cell_impl.LSTMStateTuple(num_units, num_units)
      self._output_size = num_units
    self._linear1 = None
    self._linear2 = None
    self._linear3 = None

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

  def call(self, inputs, state):
    """
    """
    (c_prev, m_prev) = state
    self._batch_size = inputs.shape[0].value or array_ops.shape(inputs)[0]
    scope = vs.get_variable_scope()
    with vs.variable_scope(scope, initializer=self._initializer):
      x = array_ops.concat([inputs, m_prev], axis=1)
      with vs.variable_scope("first_gemm"):
        if self._linear1 is None:
          self._linear1 = _Linear(x, self._fact_size, False) # no bias for bottleneck
        R_fact = self._linear1(x)
      with vs.variable_scope("second_gemm"):
        if self._linear2 is None:
          self._linear2 = _Linear(R_fact, 4*self._num_units, True)
        R = self._linear2(R_fact)
      i, j, f, o = array_ops.split(R, 4, 1)

      c = (math_ops.sigmoid(f + self._forget_bias) * c_prev +
           math_ops.sigmoid(i) * math_ops.tanh(j))
      m = math_ops.sigmoid(o) * self._activation(c)

    if self._num_proj is not None:
      with vs.variable_scope("projection"):
        if self._linear3 is None:
          self._linear3 = _Linear(m, self._num_proj, False)
        m = self._linear3(m)

    new_state = rnn_cell_impl.LSTMStateTuple(c, m)
    return m, new_state