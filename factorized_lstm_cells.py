from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import LSTMStateTuple
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _linear as linear
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.math_ops import tanh

class GLSTMCell(RNNCell):
    """LSTM cell with groups (G-LSTM) described in "FACTORIZATION TRICKS FOR LSTM NETWORKS", ICLR 2017 workshop
    https://openreview.net/pdf?id=ByxWXyNFg.
    """

    def __init__(self, num_units, initializer=None,
                 num_proj=None, number_of_groups=1,
                 forget_bias=1.0, activation=tanh):
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

        print('LSTM cell group shape: ' + str(self._group_shape))

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
            LSTM after reading `inputs` when previous state was `state`.
            Here output_dim is:
               num_proj if num_proj was set,
               num_units otherwise.
          - Tensor(s) representing the new state of LSTM after reading `inputs` when
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
        with vs.variable_scope(scope or "GLSTM_cell",
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
