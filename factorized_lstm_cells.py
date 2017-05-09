from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import LSTMStateTuple
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _linear as linear
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.util import nest
from tensorflow.python.framework import dtypes

class ResidualWrapper(RNNCell):
  """RNNCell wrapper that ensures cell inputs are added to the outputs.
  The only difference between this and TF's original wrapper is scaling
  factor of 0.5
  """

  def __init__(self, cell):
    """Constructs a `ResidualWrapper` for `cell`.
    Args:
      cell: An instance of `RNNCell`.
    """
    self._cell = cell

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def __call__(self, inputs, state, scope=None):
    """Run the cell and add its inputs to its outputs.
    Args:
      inputs: cell inputs.
      state: cell state.
      scope: optional cell scope.
    Returns:
      Tuple of cell outputs and new state.
    Raises:
      TypeError: If cell inputs and outputs have different structure (type).
      ValueError: If cell inputs and outputs have different structure (value).
    """
    outputs, new_state = self._cell(inputs, state, scope=scope)
    nest.assert_same_structure(inputs, outputs)
    # Ensure shapes match
    def assert_shape_match(inp, out):
      inp.get_shape().assert_is_compatible_with(out.get_shape())
    nest.map_structure(assert_shape_match, inputs, outputs)
    res_outputs = nest.map_structure(
        lambda inp, out: math_ops.scalar_mul(0.5, inp + out), inputs, outputs)
    return res_outputs, new_state

class GLSTMCell(RNNCell):
    """LSTM cell with groups (G-LSTM) described in "FACTORIZATION TRICKS FOR LSTM NETWORKS", ICLR 2017 workshop
    https://openreview.net/pdf?id=ByxWXyNFg.
    """

    def __init__(self, num_units,
                 initializer=None,
                 num_proj=None,
                 number_of_groups=1,
                 forget_bias=1.0,
                 activation=tanh):
        """
        Initializes parameters of G-LSTM cell
        :param num_units: int, The number of units in the G-LSTM cell
        :param initializer: (optional) The initializer to use for the weight and
            projection matrices.
        :param num_proj: (optional) int, The output dimensionality for the projection
            matrices.  If None, no projection is performed.
        :param number_of_groups: (optional) int, number of groups to use. If number_of_groups=1,
            then it should be equivalent to LSTMP cell
        :param forget_bias: Biases of the forget gate are initialized by default to 1
            in order to reduce the scale of forgetting at the beginning of
            the training.
        :param activation: Activation function of the inner states.
        """

        self._num_units = num_units
        self._initializer = initializer
        self._num_proj = num_proj
        self._forget_bias = forget_bias
        self._activation = activation
        self._number_of_groups = number_of_groups

        assert (self._num_units % self._number_of_groups == 0)
        if self._num_proj:
            assert (self._num_proj % self._number_of_groups == 0)
            self._group_shape = [self._num_proj / self._number_of_groups,
                                 self._num_units / self._number_of_groups]
        else:
            self._group_shape = [self._num_units / self._number_of_groups,
                                 self._num_units / self._number_of_groups]

        print('G-LSTM cell group shape: ' + str(self._group_shape))

        if num_proj:
            self._state_size = (LSTMStateTuple(num_units, num_proj))
            self._output_size = num_proj
        else:
            self._state_size = (LSTMStateTuple(num_units, num_units))
            self._output_size = num_units

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def _get_input_for_group(self, inpt, group_id, group_size):
        """
        Slices inputs into groups to prepare for processing by cell's groups
        :param inpt: inputs
        :param group_id: group id, for which to prepare extract input_group_id
        :param group_size: size of the group
        :return: subset of inputs, correspoinding to group group_id
        """
        return array_ops.slice(input_=inpt,
                               begin=[0, group_id * group_size],
                               size=[inpt.get_shape()[0].value, group_size],
                               name="GLSTMinputGroupCreation")

    def __call__(self, inputs, state, scope=None):
        """Run one step of G-LSTM.

        Args:
          inputs: input Tensor, 2D, batch x num_units.
          state: this must be a tuple of state Tensors, both `2-D`, with column sizes `c_state` and `m_state`.
          scope: not used

        Returns:
          A tuple containing:

          - A `2-D, [batch x output_dim]`, Tensor representing the output of the
            G-LSTM after reading `inputs` when previous state was `state`.
            Here output_dim is:
               num_proj if num_proj was set,
               num_units otherwise.
          - Tensor(s) representing the new state of G-LSTM after reading `inputs` when
            the previous state was `state`.  Same type and shape(s) as `state`.

        Raises:
          ValueError: If input size cannot be inferred from inputs via
            static shape inference.
        """
        (c_prev, m_prev) = state

        input_size = inputs.get_shape().with_rank(2)[1]
        if input_size.value is None:
            raise ValueError("Could not infer input size from inputs.get_shape()[-1]")
        dtype = inputs.dtype
        with vs.variable_scope(scope or "glstm_cell",
                               initializer=self._initializer):
            i_parts = []
            j_parts = []
            f_parts = []
            o_parts = []

            for group_id in xrange(self._number_of_groups):
                with vs.variable_scope("group%d"%group_id):
                    x_g_id = array_ops.concat([self._get_input_for_group(inputs, group_id, self._group_shape[0]),
                                               self._get_input_for_group(m_prev, group_id, self._group_shape[0])], axis=1)
                    R_k = linear(x_g_id, 4 * self._group_shape[1], bias=False, scope=scope) #will add per gate biases later
                    i_k, j_k, f_k, o_k = array_ops.split(R_k, 4, 1)

                i_parts.append(i_k)
                j_parts.append(j_k)
                f_parts.append(f_k)
                o_parts.append(o_k)

            #it is more efficient to have per gate biases then per gate, per group
            bi = vs.get_variable(name="biases_i",
                                 shape=[self._num_units],
                                 dtype=dtype,
                                 initializer=init_ops.constant_initializer(0.0, dtype=dtype))
            bj = vs.get_variable(name="biases_j",
                                 shape=[self._num_units],
                                 dtype=dtype,
                                 initializer=init_ops.constant_initializer(0.0, dtype=dtype))
            bf = vs.get_variable(name="biases_f",
                                 shape=[self._num_units],
                                 dtype=dtype,
                                 initializer=init_ops.constant_initializer(0.0, dtype=dtype))
            bo = vs.get_variable(name="biases_o",
                                 shape=[self._num_units],
                                 dtype=dtype,
                                 initializer=init_ops.constant_initializer(0.0, dtype=dtype))

            i = nn_ops.bias_add(array_ops.concat(i_parts, axis=1), bi)
            j = nn_ops.bias_add(array_ops.concat(j_parts, axis=1), bj)
            f = nn_ops.bias_add(array_ops.concat(f_parts, axis=1), bf)
            o = nn_ops.bias_add(array_ops.concat(o_parts, axis=1), bo)

        c = math_ops.sigmoid(f + self._forget_bias) * c_prev + math_ops.sigmoid(i) * math_ops.tanh(j)
        m = math_ops.sigmoid(o) * self._activation(c)

        if self._num_proj is not None:
            with vs.variable_scope("projection"):
                m = linear(m, self._num_proj, bias=False, scope=scope)

        new_state = LSTMStateTuple(c, m)
        return m, new_state

class FLSTMCell(RNNCell):
    """Factorized LSTM cell described in "FACTORIZATION TRICKS FOR LSTM NETWORKS", ICLR 2017 workshop
    https://openreview.net/pdf?id=ByxWXyNFg.
    """

    def __init__(self, num_units,
                 factor_size,
                 initializer=None,
                 num_proj=None,
                 forget_bias=1.0,
                 activation=tanh):
        """
        Initializes parameters of F-LSTM cell
        :param num_units: int, The number of units in the G-LSTM cell
        :param initializer: (optional) The initializer to use for the weight and
            projection matrices.
        :param num_proj: (optional) int, The output dimensionality for the projection
            matrices.  If None, no projection is performed.
        :param factor_size: factorization size
        :param forget_bias: Biases of the forget gate are initialized by default to 1
            in order to reduce the scale of forgetting at the beginning of
            the training.
        :param activation: Activation function of the inner states.
        """

        self._num_units = num_units
        self._initializer = initializer
        self._num_proj = num_proj
        self._forget_bias = forget_bias
        self._activation = activation
        self._factor_size = factor_size

        assert (self._num_units > self._factor_size)
        if self._num_proj:
            assert (self._num_proj > self._factor_size)

        if num_proj:
            self._state_size = (LSTMStateTuple(num_units, num_proj))
            self._output_size = num_proj
        else:
            self._state_size = (LSTMStateTuple(num_units, num_units))
            self._output_size = num_units

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def __call__(self, inputs, state, scope=None):
        """Run one step of F-LSTM.

        Args:
          inputs: input Tensor, 2D, batch x num_units.
          state: this must be a tuple of state Tensors, both `2-D`, with column sizes `c_state` and `m_state`.
          scope: not used

        Returns:
          A tuple containing:

          - A `2-D, [batch x output_dim]`, Tensor representing the output of the
            F-LSTM after reading `inputs` when previous state was `state`.
            Here output_dim is:
               num_proj if num_proj was set,
               num_units otherwise.
          - Tensor(s) representing the new state of F-LSTM after reading `inputs` when
            the previous state was `state`.  Same type and shape(s) as `state`.

        Raises:
          ValueError: If input size cannot be inferred from inputs via
            static shape inference.
        """
        (c_prev, m_prev) = state

        input_size = inputs.get_shape().with_rank(2)[1]
        if input_size.value is None:
            raise ValueError("Could not infer input size from inputs.get_shape()[-1]")
        with vs.variable_scope(scope or "flstm_cell",
                               initializer=self._initializer):
            with vs.variable_scope("factor"):
                fact = linear([inputs, m_prev], self._factor_size, False)
            concat = linear(fact, 4 * self._num_units, True)
            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = array_ops.split(value=concat, num_or_size_splits=4, axis=1)


        c = math_ops.sigmoid(f + self._forget_bias) * c_prev + math_ops.sigmoid(i) * math_ops.tanh(j)
        m = math_ops.sigmoid(o) * self._activation(c)

        if self._num_proj is not None:
            with vs.variable_scope("projection"):
                m = linear(m, self._num_proj, bias=False, scope=scope)

        new_state = LSTMStateTuple(c, m)
        return m, new_state


class XLSTMCell(RNNCell):
  """
  """

  def __init__(self, num_units,
               initializer=None,
               num_proj=None,
               forget_bias=1.0,
               activation=tanh):
    """
    Initializes parameters of F-LSTM cell
    :param num_units: int, The number of units in the G-LSTM cell
    :param initializer: (optional) The initializer to use for the weight and
        projection matrices.
    :param num_proj: (optional) int, The output dimensionality for the projection
        matrices.  If None, no projection is performed.
    :param factor_size: factorization size
    :param forget_bias: Biases of the forget gate are initialized by default to 1
        in order to reduce the scale of forgetting at the beginning of
        the training.
    :param activation: Activation function of the inner states.
    """

    self._num_units = num_units
    self._initializer = initializer
    self._num_proj = num_proj
    self._forget_bias = forget_bias
    self._activation = activation

    if num_proj:
      self._state_size = (LSTMStateTuple(num_units, num_proj))
      self._output_size = num_proj
    else:
      self._state_size = (LSTMStateTuple(num_units, num_units))
      self._output_size = num_units

    scope = vs.get_variable_scope()
    with vs.variable_scope(scope) as outer_scope:
      self.weights32 = vs.get_variable(
        "weights32", [2*num_proj, 4*num_units], dtype=dtypes.float32)
      self.biases32 = vs.get_variable(
          "biases", [4*num_units],
          dtype=dtypes.float32,
          initializer=init_ops.constant_initializer(0.0, dtype=dtypes.float32))
      self.proj32 = vs.get_variable(
        "projection32", [num_units, num_proj], dtype=dtypes.float32)

      self.weights = math_ops.cast(self.weights32, dtypes.float16)
      self.biases = math_ops.cast(self.biases32, dtypes.float16)
      self.proj = math_ops.cast(self.proj32, dtype=dtypes.float16)

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

  def __call__(self, inputs, state, scope=None):
    """Run one step of F-LSTM.

    Args:
      inputs: input Tensor, 2D, batch x num_units.
      state: this must be a tuple of state Tensors, both `2-D`, with column sizes `c_state` and `m_state`.
      scope: not used

    Returns:
      A tuple containing:

      - A `2-D, [batch x output_dim]`, Tensor representing the output of the
        F-LSTM after reading `inputs` when previous state was `state`.
        Here output_dim is:
           num_proj if num_proj was set,
           num_units otherwise.
      - Tensor(s) representing the new state of F-LSTM after reading `inputs` when
        the previous state was `state`.  Same type and shape(s) as `state`.

    Raises:
      ValueError: If input size cannot be inferred from inputs via
        static shape inference.
    """
    (c_prev, m_prev) = state

    input_size = inputs.get_shape().with_rank(2)[1]
    if input_size.value is None:
      raise ValueError("Could not infer input size from inputs.get_shape()[-1]")
    with vs.variable_scope(scope or "xlstm_cell",
                           initializer=self._initializer):
      #concat = Xlinear([inputs, m_prev], 4 * self._num_units, True)
      concat = nn_ops.xw_plus_b(array_ops.concat([inputs, m_prev], 1), self.weights, self.biases)
      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      i, j, f, o = array_ops.split(value=concat, num_or_size_splits=4, axis=1)

    c = math_ops.sigmoid(f + self._forget_bias) * c_prev + math_ops.sigmoid(i) * math_ops.tanh(j)
    m = math_ops.sigmoid(o) * self._activation(c)

    if self._num_proj is not None:
      with vs.variable_scope("projection"):
        #m = Xlinear(m, self._num_proj, bias=False, scope=scope)
        m = math_ops.matmul(m, self.proj)

    new_state = LSTMStateTuple(c, m)
    return m, new_state