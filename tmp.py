# now the LSTMs
# these will collect the initial states for the forward
#   (and reverse LSTMs if we are doing bidirectional)
self.init_lstm_state = []
self.final_lstm_state = []

# get the LSTM inputs
if self.bidirectional:
    rnn_input = [self.embedding, self.embedding_reverse]
else:
    rnn_input = [self.embedding]

# now compute the LSTM outputs
cell_clip = self.options['lstm'].get('cell_clip')
proj_clip = self.options['lstm'].get('proj_clip')

rim = self.options['lstm'].get('rim')
if rim:
    top_k = rim.get('top_k')
    n_rim = rim.get('n_rim')

    num_input_heads = rim.get('num_input_heads')
    input_key_size = rim.get('input_key_size')
    input_value_size = rim.get('input_value_size')
    input_query_size = rim.get('input_query_size')
    input_keep_prob = rim.get('input_keep_prob')

    num_comm_heads=rim.get('num_comm_heads')
    comm_key_size=rim.get('comm_key_size')
    comm_value_size=rim.get('comm_value_size')
    comm_query_size=rim.get('comm_query_size')
    comm_keep_prob=rim.get('comm_keep_prob')

use_skip_connections = self.options['lstm'].get(
                                    'use_skip_connections')
if use_skip_connections:
    print("USING SKIP CONNECTIONS")

rim_outputs = []
for rnn_num, rnn_input in enumerate(rnn_input):
    if rim:
        rim_cells=[]
    else:
        lstm_cells = []
    for i in range(n_lstm_layers):
        if rim:
            rim_cell = RIMCell(units = lstm_dim, nRIM=n_rim, k=top_k,
                num_input_heads=num_input_heads, input_key_size=input_key_size, input_value_size=input_value_size, input_query_size=input_query_size, input_keep_prob=input_keep_prob,
                num_comm_heads=num_comm_heads, comm_key_size=comm_key_size, comm_value_size=comm_value_size, comm_query_size=comm_query_size, comm_keep_prob=comm_keep_prob)
        else:
            if projection_dim < lstm_dim:
                # are projecting down output
                lstm_cell = tf.nn.rnn_cell.LSTMCell(
                    lstm_dim, num_proj=projection_dim,
                    cell_clip=cell_clip, proj_clip=proj_clip)
            else:
                lstm_cell = tf.nn.rnn_cell.LSTMCell(
                    lstm_dim,
                    cell_clip=cell_clip, proj_clip=proj_clip)

        if use_skip_connections:
            # ResidualWrapper adds inputs to outputs
            if i == 0:
                # don't add skip connection from token embedding to
                # 1st layer output
                pass
            else:
                if rim:
                    rim_cell = ResidualRNNCells(rim_cell)
                else:
                    # add a skip connection
                    lstm_cell = tf.nn.rnn_cell.ResidualWrapper(lstm_cell)

        # add dropout
        if self.is_training:
            if rim:
                pass
            else:
                lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,
                    input_keep_prob=keep_prob)

        if rim:
            rim_cells.append(rim_cell)
        else:
            lstm_cells.append(lstm_cell)

    if n_lstm_layers > 1:
        if rim:
            rim_cell = tf.keras.layers.StackedRNNCells(cells)
        else:
            lstm_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)
    else:
        if rim:
            rim_cell = rim_cells[0]
        else:
            lstm_cell = lstm_cells[0]

    if rim:
        with tf.control_dependencies([rnn_input]):
            self.init_rim_state.append(
                rim_cell.get_initial_state(batch_size, DTYPE))
            _lstm_output_unpacked, final_state = rim_cell(
                tf.unstack(rnn_input, axis=1),
                self.init_rim_state[-1]
            )
            self.final_lstm_state.append(final_state)
    else:
        with tf.control_dependencies([rnn_input]):
            self.init_lstm_state.append(
                lstm_cell.zero_state(inputs=None, batch_size=batch_size, dtype=DTYPE))
            # NOTE: this variable scope is for backward compatibility
            # with existing models...
            if self.bidirectional:
                with tf.variable_scope('RNN_%s' % rnn_num):
                    _lstm_output_unpacked, final_state = tf.nn.static_rnn(
                        lstm_cell,
                        tf.unstack(rnn_input, axis=1),
                        initial_state=self.init_lstm_state[-1])
            else:
                _lstm_output_unpacked, final_state = tf.nn.static_rnn(
                    lstm_cell,
                    tf.unstack(rnn_input, axis=1),
                    initial_state=self.init_lstm_state[-1])
            self.final_lstm_state.append(final_state)

    # (batch_size * unroll_steps, 512)
    lstm_output_flat = tf.reshape(
        tf.stack(_lstm_output_unpacked, axis=1), [-1, projection_dim])
    if self.is_training:
        # add dropout to output
        lstm_output_flat = tf.nn.dropout(lstm_output_flat,
            keep_prob)
    tf.add_to_collection('lstm_output_embeddings',
        _lstm_output_unpacked)

    rim_outputs.append(lstm_output_flat)