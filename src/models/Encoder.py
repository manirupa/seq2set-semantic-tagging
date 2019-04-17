from abc import ABC, abstractmethod

import tensorflow as tf
from tensor2tensor.layers.transformer_layers import transformer_prepare_encoder, transformer_encoder
from tensor2tensor.models.transformer import transformer_base


class EncoderFactory:
    @staticmethod
    def get_encoder(params):
        if params['model'] == "DAN":
            return _DAN(params)
        if params['model'] == "LSTM":
            return _LSTM(params)
        if params['model'] == "GRU":
            return _GRU(params)
        if params['model'] == "Transformer":
            return _TransformerEncoder(params)
        if params['model'] == "BiLSTMATT":
            return _BiLSTMATT(params)
        if params['model'] == "BiLSTM":
            return _BiLSTM(params)
        if params['model'] == "BiGRU":
            return _BiGRU(params)
        else:
            raise ValueError('Invalid model name')


class _Encoder(ABC):
    def __init__(self, params):
        self.mlp_layer_dims = params['mlp_layer_dims']
        self.doc_vec_length = params['doc_vec_length']
        self.dropout = params['dropout']

    def encode(self, x, sen_len):
        x = self.model(x, sen_len)
        doc_vecs, logits = self.mlp(x)
        return doc_vecs, logits

    @abstractmethod
    def model(self, x, sen_len):
        pass

    def mlp(self, x):
        with tf.name_scope("mlp"):
            doc_vec = None
            for i in range(len(self.mlp_layer_dims)):
                x = tf.layers.dense(
                    x, units=self.mlp_layer_dims[i],
                    kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                    bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                    activation=tf.nn.leaky_relu)
                if self.mlp_layer_dims[i] == self.doc_vec_length:
                    doc_vec = tf.identity(x, name='doc_vec')

        with tf.name_scope("dropout"):  # self.dropout should be 0 during inference!
            x = tf.nn.dropout(x, 1.0 - self.dropout)
        return doc_vec, x

    @staticmethod
    def reduce_sequence_dim(x):
        with tf.name_scope("reduce_sequence_dim"):
            x = tf.layers.dense(
                x, units=1,
                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                activation=tf.nn.leaky_relu)

            x = tf.squeeze(x, axis=2)
        return x


class _DAN(_Encoder):
    def model(self, x, sen_len):
        with tf.name_scope("average"):
            x = tf.reduce_mean(x, axis=1, name='average')
        return x


class _LSTM(_Encoder):
    def __init__(self, params):
        super(_LSTM, self).__init__(params)
        self.num_layers = params['num_layers']
        self.hidden_size = params['hidden_size']

    def model(self, x, sen_len):
        with tf.name_scope("lstm"):
            # reshape embed to shape [sequence_len, batch_size, embedding_dim]
            x = tf.transpose(x, [1, 0, 2])
            lstm = tf.contrib.cudnn_rnn.CudnnLSTM(
                num_layers=self.num_layers,
                num_units=self.hidden_size, name="lstm")
            _, (output_states_h, _) = lstm(x)
        return output_states_h[-1]  # TODO: try pooling


class _GRU(_Encoder):
    def __init__(self, params):
        super(_GRU, self).__init__(params)
        self.num_layers = params['num_layers']
        self.hidden_size = params['hidden_size']

    def model(self, x, sen_len):
        with tf.name_scope("gru"):
            gru = tf.keras.layers.GRU(
                self.hidden_size, return_sequences=False, return_state=True)
            _, output_states_h = gru(x)
        return output_states_h  # TODO: try pooling


class _TransformerEncoder(_Encoder):
    def __init__(self, params):
        super(_TransformerEncoder, self).__init__(params)
        self.hparams = transformer_base()
        self.hparams.hidden_size = params['embeddings_dim']
        self.hparams.num_heads = params['num_heads']
        self.hparams.num_hidden_layers = params['num_layers']
        self.hparams.pos = None
        self.hparams.use_target_space_embedding = False
        # self.hparams.layer_prepostprocess_dropout = 0.3

        # A scalar int from data_generators.problem.SpaceID.
        # 0 represents generic / unknown output space (default)
        self.target_space = 0

    def model(self, x, sen_len):
        x, enc_self_att_bias, _ = transformer_prepare_encoder(
            x, self.target_space, self.hparams)
        # x shape (batch_size, sequence_len, hidden_size)
        # enc_self_att_bias shape (batch_size, 1, 1, sequence_len)

        x = transformer_encoder(x, enc_self_att_bias, self.hparams)
        # shape (batch_size, sequence_length, hidden_size)

        indices = tf.stack([tf.range(0, tf.shape(x)[0]),
                            sen_len - 1],  # the last indices of each sentence
                           axis=1)

        final_output_state = tf.gather_nd(x, indices, name="select_last_word_in_sentence")
        # shape: (batch_size, hidden_size)

        return final_output_state


class _BiLSTM(_LSTM):
    def bi_lstm(self, x, sequence_length):
        with tf.name_scope("bi_lstm"):
            cells_fw = [self.single_cell() for _ in range(self.num_layers)]
            cells_bw = [self.single_cell() for _ in range(self.num_layers)]

            x, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                cells_fw, cells_bw, x, dtype=tf.float32, sequence_length=sequence_length)
            # shape of x: (batch_size, sequence_length, embed_dim*2)
            # shape of output_state_fw, bw: [output_state_layer_0, output_state_layer_1, ...]
            # shape of output_state_layer_i: (cell_state, hidden_state)
            # shape of cell_state, hidden_state: [batch_size, hidden_size]
        return x

    def model(self, x, sequence_length):
        x = self.bi_lstm(x, sequence_length)

        indices = tf.stack([tf.range(0, tf.shape(x)[0]),
                            sequence_length - 1],  # the last indices of each sentence
                           axis=1)

        final_output_state = tf.gather_nd(x, indices, name="select_last_word_in_sentence")
        # shape: (batch_size, embed_dim*2)

        return final_output_state

    def single_cell(self):
        return tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(self.hidden_size)


class _BiGRU(_GRU):
    def bi_gru(self, x, sequence_length):
        with tf.name_scope("bi_gru"):
            cells_fw = [self.single_cell() for _ in range(self.num_layers)]
            cells_bw = [self.single_cell() for _ in range(self.num_layers)]

            x, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                cells_fw, cells_bw, x, dtype=tf.float32, sequence_length=sequence_length)
            # shape of x: (batch_size, sequence_length, embed_dim*2)
            # shape of output_state_fw, bw: [output_state_layer_0, output_state_layer_1, ...]
            # shape of output_state_layer_i: (cell_state, hidden_state)
            # shape of cell_state, hidden_state: [batch_size, hidden_size]
        return x

    def model(self, x, sequence_length):
        x = self.bi_gru(x, sequence_length)

        indices = tf.stack([tf.range(0, tf.shape(x)[0]),
                            sequence_length - 1],  # the last indices of each sentence
                           axis=1)

        final_output_state = tf.gather_nd(x, indices, name="select_last_word_in_sentence")
        # shape: (batch_size, embed_dim*2)

        return final_output_state

    def single_cell(self):
        return tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(self.hidden_size)


class _BiLSTMATT(_BiLSTM):
    def __init__(self, params):
        super(_BiLSTMATT, self).__init__(params)
        self.attention_size = params['attention_size']

    def model(self, x, sen_len):
        x = self.bi_lstm(x, sen_len)
        with tf.name_scope("attention"):
            x = self.attention(x, self.attention_size)
        return x

    @staticmethod
    def attention(inputs, attention_size, time_major=False, return_alphas=False):
        """
        Attention mechanism layer which reduces RNN/Bi-RNN
        outputs with Attention vector. The idea was proposed
        in the article by Z. Yang et al., "Hierarchical
        Attention Networks for Document Classification", 2016:
        http://www.aclweb.org/anthology/N16-1174. Variables
        notation is also inherited from the article

        Args:
            inputs: The Attention inputs.
                Matches outputs of RNN/Bi-RNN layer (not final state):
                    In case of RNN, this must be RNN outputs `Tensor`:
                        If time_major == False (default), this must be
                        a tensor of shape:
                            `[batch_size, max_time, cell.output_size]`.
                        If time_major == True, this must be a tensor
                        of shape:
                            `[max_time, batch_size, cell.output_size]`.
                    In case of Bidirectional RNN, this must be a tuple
                    (outputs_fw, outputs_bw) containing the forward and
                    the backward RNN outputs `Tensor`.
                        If time_major == False (default),
                            outputs_fw is a `Tensor` shaped:
                            `[batch_size, max_time, cell_fw.output_size]`
                            and outputs_bw is a `Tensor` shaped:
                            `[batch_size, max_time, cell_bw.output_size]`.
                        If time_major == True,
                            outputs_fw is a `Tensor` shaped:
                            `[max_time, batch_size, cell_fw.output_size]`
                            and outputs_bw is a `Tensor` shaped:
                            `[max_time, batch_size, cell_bw.output_size]`.
            attention_size: Linear size of the Attention weights.
            time_major: The shape format of the `inputs` Tensors.
                If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
                If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
                Using `time_major = True` is a bit more efficient because it avoids
                transposes at the beginning and end of the RNN calculation.  However,
                most TensorFlow data is batch-major, so by default this function
                accepts input and emits output in batch-major form.
            return_alphas: Whether to return attention coefficients
            variable along with layer's output.
                Used for visualization purpose.
        Returns:
            The Attention output `Tensor`.
            In case of RNN, this will be a `Tensor` shaped:
                `[batch_size, cell.output_size]`.
            In case of Bidirectional RNN, this will be a `Tensor` shaped:
                `[batch_size, cell_fw.output_size + cell_bw.output_size]`.
        """

        if isinstance(inputs, tuple):
            # In case of Bi-RNN, concatenate the forward
            # and the backward RNN outputs.
            inputs = tf.concat(inputs, 2)

        if time_major:
            # (T,B,D) => (B,T,D)
            inputs = tf.transpose(inputs, [1, 0, 2])

        # D value - hidden size of the RNN layer
        hidden_size = inputs.shape[2].value

        # Trainable parameters
        w_omega = tf.Variable(tf.random_normal(
            [hidden_size, attention_size], stddev=0.1))
        b_omega = tf.Variable(tf.random_normal(
            [attention_size], stddev=0.1))
        u_omega = tf.Variable(tf.random_normal(
            [attention_size], stddev=0.1))

        with tf.name_scope('v'):
            # Applying fully connected layer with non-linear
            # activation to each of the B*T timestamps;
            #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A),
            # where A=attention_size
            v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

        # For each of the timestamps its vector of size A
        # from `v` is reduced with `u` vector
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
        alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

        # Output of (Bi-)RNN is reduced with attention vector;
        # the result has (B,D) shape
        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

        if not return_alphas:
            return output
        else:
            return output, alphas
