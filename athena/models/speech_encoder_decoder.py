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
r""" some implementations for seq2seq models

such as the NeuralTranslate, an example of seq2seq model; NeuralTranslateTransformer,
an example of seq2seq model using transformer
"""

import tensorflow as tf
from absl import logging
from .base import BaseModel
from ..loss import Seq2SeqSparseCategoricalCrossentropy
from ..metrics import Seq2SeqSparseCategoricalAccuracy
from ..layers.attention_decoder import AttentionDecoder
from ..layers.commons import SUPPORTED_RNNS
from ..utils.misc import insert_sos_in_labels
from ..utils.hparam import register_and_parse_hparams

#pylint: disable=abstract-method, no-member, too-many-instance-attributes
class ListenAttendSpell(BaseModel):
    ''' This is an example of seq2seq model. '''
    default_config = {
        "return_encoder_output": False,
        "conv_filters": 512,
        "embedding_dim": 512,
        "d_model": 512,
        "rnn_type": "gru",
        "num_encoder_rnn_layers": 6
    }
    #pylint: disable=invalid-name
    def __init__(self, num_classes, sample_shape, config=None):
        ''' init function '''
        super().__init__()
        self.num_classes = num_classes + 1
        self.sos = num_classes
        self.eos = num_classes
        self.hparams = register_and_parse_hparams(self.default_config, config, cls=self.__class__)
        p = self.hparams
        self.loss_function = Seq2SeqSparseCategoricalCrossentropy(
            num_classes=self.num_classes,
            eos=self.eos,
            label_smoothing=p.label_smoothing_rate
        )
        self.metric = Seq2SeqSparseCategoricalAccuracy(
            eos=self.eos,
            name='Accuracy'
        )

        layers = tf.keras.layers
        input_feature = layers.Input(shape=sample_shape['input'], dtype=tf.float32)
        inner = layers.Conv2D(
            filters=p.conv_filters,
            kernel_size=(5, 5),
            strides=(2, 2),
            padding="same",
            use_bias=False,
        )(input_feature)
        inner = layers.BatchNormalization()(inner)
        inner = tf.nn.relu6(inner)
        inner = layers.Conv2D(
            filters=p.conv_filters,
            kernel_size=(5, 5),
            strides=(2, 2),
            padding="same",
            use_bias=False
        )(inner)
        inner = layers.BatchNormalization()(inner)
        inner = tf.nn.relu6(inner)
        _, _, dim, channels = inner.get_shape().as_list()
        output_dim = dim * channels
        inner = layers.Reshape((-1, output_dim))(inner)

        for _ in range(p.num_encoder_rnn_layers - 1):
            inner = SUPPORTED_RNNS[p.rnn_type](
                p.d_model,
                return_sequences=True
            )(inner)
            inner = layers.BatchNormalization()(inner)
        self.x_net = tf.keras.Model(
            inputs=input_feature,
            outputs=inner,
            name='x_net')
        logging.info(self.x_net.summary())

        self.rnn = SUPPORTED_RNNS[p.rnn_type](
            p.d_model,
            input_shape=[None, p.d_model],
            return_sequences=True,
            return_state=True
        )
        self.norm = layers.BatchNormalization()
        self.decoder = AttentionDecoder(
            self.num_classes,
            p.embedding_dim,
            p.d_model
        )

    #pylint: disable=invalid-name
    def call(self, samples, training=None):
        x = samples['input']
        y = insert_sos_in_labels(samples['output'], self.sos)

        x = self.x_net(x)
        x, m = self.rnn(x)
        x = self.norm(x)
        y = self.decoder(x, m, y, training=training)

        if self.hparams.return_encoder_output:
            return y, x
        return y

    def compute_logit_length(self, samples):
        logit_length = tf.cast(samples['input_length'], tf.float32)
        logit_length = tf.math.ceil(logit_length / 2)
        logit_length = tf.math.ceil(logit_length / 2)
        logit_length = tf.cast(logit_length, tf.int32)
        return logit_length
