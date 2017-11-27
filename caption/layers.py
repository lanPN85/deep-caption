from keras.engine import InputSpec, Layer
from keras.layers import LSTMCell, RNN, regularizers, initializers

import keras.backend as K
import tensorflow as tf

assert K.backend() == 'tensorflow'


def decode_rnn(step_function, inputs, initial_states, timesteps):
    ndim = len(inputs.get_shape())
    # if ndim < 3:
    #     raise ValueError('Input should be at least 3D.')
    constants = []

    # Transpose to time-major, i.e.
    # from (batch, time, ...) to (time, batch, ...)
    # axes = [1, 0] + list(range(2, ndim))
    # inputs = tf.transpose(inputs, axes)

    states = tuple(initial_states)

    time_steps = timesteps
    outputs, _ = step_function(inputs, initial_states + constants)
    output_ta = K.tensor_array_ops.TensorArray(
        dtype=outputs.dtype,
        size=time_steps,
        tensor_array_name='output_ta')
    # input_ta = K.tensor_array_ops.TensorArray(
    #     dtype=inputs.dtype,
    #     size=time_steps,
    #     tensor_array_name='input_ta')
    time = K.constant(0, dtype='int32', name='time')

    # input_ta = input_ta.write(time, inputs)

    def _step(time, output_ta_t, input_t, *states):
        """RNN decoder step function.

        # Arguments
            time: Current timestep value.
            output_ta_t: TensorArray.
            *states: List of states.

        # Returns
            Tuple: `(time + 1,output_ta_t) + tuple(new_states)`
        """
        # current_input = input_ta_t.read(time)
        current_input = input_t
        output, _new_states = step_function(current_input,
                                            tuple(states) +
                                            tuple(constants))
        for state, new_state in zip(states, _new_states):
            new_state.set_shape(state.get_shape())
        output_ta_t = output_ta_t.write(time, output)
        input_t = K.switch(time < time_steps - K.constant(1, dtype='int32'),
                           output, input_t) # Write output to next input
        return (time + 1, output_ta_t, input_t) + tuple(_new_states)

    final_outputs = K.control_flow_ops.while_loop(
        cond=lambda _time, *_: _time < time_steps,
        body=_step,
        loop_vars=(time, output_ta, inputs) + states,
        parallel_iterations=32,
        swap_memory=True)
    last_time = final_outputs[0]
    output_ta = final_outputs[1]
    new_states = final_outputs[3:]

    outputs = output_ta.stack()
    last_output = output_ta.read(last_time - 1)

    axes = [1, 0] + list(range(2, len(outputs.get_shape())))
    outputs = tf.transpose(outputs, axes)
    # last_output._uses_learning_phase = uses_learning_phase
    return last_output, outputs, new_states


# noinspection PyAttributeOutsideInit
class DecoderLSTMCell(LSTMCell):
    def __init__(self, units, **kwargs):
        super().__init__(units, **kwargs)

    def build(self, input_shape):
        input_dim = input_shape
        self.kernel = self.add_weight(shape=(input_dim, self.units * 4),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        if self.use_bias:
            if self.unit_forget_bias:
                def bias_initializer(shape, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.units,), *args, **kwargs),
                        initializers.Ones()((self.units,), *args, **kwargs),
                        self.bias_initializer((self.units * 2,), *args, **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(shape=(self.units * 4,),
                                        name='bias',
                                        initializer=bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.kernel_i = self.kernel[:, :self.units]
        self.kernel_f = self.kernel[:, self.units: self.units * 2]
        self.kernel_c = self.kernel[:, self.units * 2: self.units * 3]
        self.kernel_o = self.kernel[:, self.units * 3:]

        self.recurrent_kernel_i = self.recurrent_kernel[:, :self.units]
        self.recurrent_kernel_f = self.recurrent_kernel[:, self.units: self.units * 2]
        self.recurrent_kernel_c = self.recurrent_kernel[:, self.units * 2: self.units * 3]
        self.recurrent_kernel_o = self.recurrent_kernel[:, self.units * 3:]

        if self.use_bias:
            self.bias_i = self.bias[:self.units]
            self.bias_f = self.bias[self.units: self.units * 2]
            self.bias_c = self.bias[self.units * 2: self.units * 3]
            self.bias_o = self.bias[self.units * 3:]
        else:
            self.bias_i = None
            self.bias_f = None
            self.bias_c = None
            self.bias_o = None
        self.built = True

    def call(self, inputs, states, training=None):
        # dropout matrices for input units
        dp_mask = self._dropout_mask
        # dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask

        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        if 0. < self.dropout < 1.:
            inputs *= dp_mask[0]
        z = K.dot(inputs, self.kernel)
        if 0. < self.recurrent_dropout < 1.:
            h_tm1 *= rec_dp_mask[0]
        z += K.dot(h_tm1, self.recurrent_kernel)
        if self.use_bias:
            z = K.bias_add(z, self.bias)

        z0 = z[:, :self.units]
        z1 = z[:, self.units: 2 * self.units]
        z2 = z[:, 2 * self.units: 3 * self.units]
        z3 = z[:, 3 * self.units:]

        i = self.recurrent_activation(z0)
        f = self.recurrent_activation(z1)
        c = f * c_tm1 + i * self.activation(z2)
        o = self.recurrent_activation(z3)

        h = o * self.activation(c)
        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                h._uses_learning_phase = True
        return h, [h, c]


class DecoderLSTM(RNN):
    def __init__(self, units, output_length,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 return_state=False,
                 go_backwards=False,
                 unroll=False,
                 **kwargs):
        cell = DecoderLSTMCell(units,
                               activation=activation,
                               recurrent_activation=recurrent_activation,
                               use_bias=use_bias,
                               kernel_initializer=kernel_initializer,
                               recurrent_initializer=recurrent_initializer,
                               unit_forget_bias=unit_forget_bias,
                               bias_initializer=bias_initializer,
                               kernel_regularizer=kernel_regularizer,
                               recurrent_regularizer=recurrent_regularizer,
                               bias_regularizer=bias_regularizer,
                               kernel_constraint=kernel_constraint,
                               recurrent_constraint=recurrent_constraint,
                               bias_constraint=bias_constraint,
                               dropout=dropout,
                               recurrent_dropout=recurrent_dropout,
                               implementation=implementation
                               )
        super().__init__(cell,
                         return_sequences=True,
                         return_state=return_state,
                         go_backwards=go_backwards,
                         stateful=False,
                         unroll=unroll,
                         **kwargs
                         )

        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.output_length = output_length
        self.input_spec = [InputSpec(ndim=2)]

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        batch_size = None
        input_dim = input_shape[-1]
        self.input_spec[0] = InputSpec(shape=(batch_size, input_dim))

        # allow cell (if layer) to build before we set or validate state_spec
        if isinstance(self.cell, Layer):
            step_input_shape = input_shape[-1]
            self.cell.build(step_input_shape)

        # set or validate state_spec
        if hasattr(self.cell.state_size, '__len__'):
            state_size = list(self.cell.state_size)
        else:
            state_size = [self.cell.state_size]

        if self.state_spec is not None:
            # initial_state was passed in call, check compatibility
            if not [spec.shape[-1] for spec in self.state_spec] == state_size:
                raise ValueError(
                    'An initial_state was passed that is not compatible with '
                    '`cell.state_size`. Received `state_spec`={}; '
                    'However `cell.state_size` is '
                    '{}'.format(self.state_spec, self.cell.state_size))
        else:
            self.state_spec = [InputSpec(shape=(None, dim))
                               for dim in state_size]

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        return input_shape[0], self.output_length, self.cell.units

    def call(self,
             inputs,
             mask=None,
             training=None,
             initial_state=None,
             constants=None):
        if isinstance(inputs, list):
            inputs = inputs[0]
        if initial_state is not None:
            pass
        else:
            initial_state = self.get_initial_state(inputs)

        if len(initial_state) != len(self.states):
            raise ValueError('Layer has ' + str(len(self.states)) +
                             ' states but was passed ' +
                             str(len(initial_state)) +
                             ' initial states.')
        input_shape = K.int_shape(inputs)
        timesteps = self.output_length

        kwargs = {}
        if K.has_arg(self.cell.call, 'training'):
            kwargs['training'] = training

        def step(inputs, states):
            return self.cell.call(inputs, states, **kwargs)

        last_output, outputs, states = decode_rnn(step, inputs, initial_state,
                                                  timesteps=timesteps)

        if self.return_state:
            if not isinstance(states, (list, tuple)):
                states = [states]
            else:
                states = list(states)
            return [outputs] + states
        else:
            return outputs

    def get_initial_state(self, inputs):
        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = K.zeros_like(inputs)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=(1,))  # (samples,)
        initial_state = K.expand_dims(initial_state)  # (samples, 1)
        if hasattr(self.cell.state_size, '__len__'):
            return [K.tile(initial_state, [1, dim])
                    for dim in self.cell.state_size]
        else:
            return [K.tile(initial_state, [1, self.cell.state_size])]
