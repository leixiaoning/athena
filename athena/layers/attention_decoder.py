# coding=utf-8
# Copyright (C) ATHENA AUTHORS
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Only support eager mode
# pylint: disable=relative-beyond-top-level, too-many-arguments, invalid-name
""" the decoder layer in encoder-decoder models """
import tensorflow as tf
from .attention import BahdanauAttention
from .commons import SUPPORTED_RNNS

class AttentionDecoder(tf.keras.layers.Layer):
    r""" Used in Encoder-Decoder Models, specifical in the Listen-Attend-Spell
    Reference in arXiv:1508.01211

    Args:
        vocab_size: the size of the vocab, also the dimension of output layer
        embedding_dim: the dimension of the embedding layer used for vocab
        start: the representation of the start symbol, should be a id
        d_model: the number of expected size of the hidden layers
        rnn_type: the rnn type, can be gru or lstm, (default='gru')
        custom_decoder: we can custom a decoder (default=None)

    Examples::
        >>> decoder = AttentionDecoder(1024, start=1023, embedding_dim=256, d_model=1024)
        >>> x = tf.random.normal((2, 3, 4))
        >>> memory = tf.random.normal((2, 4))
        >>> y = tf.random.normal((2, 3), dtype=tf.dtypes.int32)
        >>> out = decoder(x, memory, y)
    """
    def __init__(
        self,
        num_classes,
        embedding_dim,
        d_model,
        rnn_type='gru',
        custom_decoder=None
    ):
        super().__init__()
        layers = tf.keras.layers
        self.d_model = d_model
        # embedding
        self.embedding = layers.Embedding(num_classes, embedding_dim)
        self.attention = BahdanauAttention(self.d_model)
        if custom_decoder is not None:
            self.rnn = custom_decoder
        else:
            self.rnn = SUPPORTED_RNNS[rnn_type](
                d_model,
                return_sequences=True,
                return_state=True,
                recurrent_initializer='glorot_uniform'
            )
        self.dense = layers.Dense(num_classes)

    def call(self, x, m, y, training=None):
        r""" Take in and process target sequences

        Args:
            x: the output of encoder layers, the value for attention
            m: usually the state output of encoder, the query for attention
            y: the sequence to the decoder
        Shape:
            - x: [N, T, D]
            - memory: [N, D]
            - y: [N, T]
        """
        memory = m
        outputs = tf.TensorArray(
            tf.float32,
            size=tf.shape(y)[1] - 1,
            dynamic_size=True
        )
        for t in tf.range(0, tf.shape(y)[1]):
            inner = tf.expand_dims(y[:, t], 1)
            output, memory, _ = self.time_propagate(inner, memory, x, training=training)
            outputs = outputs.write(t, output)
        return tf.transpose(outputs.stack(), [1, 0, 2])


    def time_propagate(self, y, memory, enc_output, training=None):
        r""" Take in and process target sequences only propagate 1 time step

        Args:
            y: the sequence to the decoder
            memory: usually the state output of encoder, the query for attention
            enc_output: the output of encoder layers, the value for attention
        Shape:
            - y: [N, T]
            - memory: [N, D]
            - enc_output: [N, T, D]
        Returns:
            output: the output layer's output in 1 time step
            memory: the output of decoder rnn state
            weight: the attention weights
        """
        context, weight = self.attention(memory, enc_output, training=training)
        y = self.embedding(y, training=training)
        y = tf.concat([tf.expand_dims(context, 1), y], axis=-1)
        output, memory = self.rnn(y, training=training)
        # output, memory = self.rnn(y, initial_state=memory, training=training)
        output = tf.reshape(output, (-1, output.shape[2]))
        output.set_shape([None, self.d_model])
        output = self.dense(output, training=training)
        return output, memory, weight
