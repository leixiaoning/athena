# Copyright (C) 2017 Beijing Didi Infinity Technology and Development Co.,Ltd.
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

import tensorflow as tf

from athena.utils.hparam import HParams
from athena.transform.feats.ops import py_x_ops
from athena.transform.feats.base_frontend import BaseFrontend
from athena.transform.feats.spectrum import Spectrum
from athena.transform.feats.cmvn import CMVN


class Fbank(BaseFrontend):
    def __init__(self, config: dict):
        super().__init__(config)
        self.spect = Spectrum(config)
        self.cmvn = CMVN(config)

        # global cmvn dim == feature dim
        if config.type == "Fbank" and self.cmvn.global_cmvn:
            assert config.filterbank_channel_count * config.channel == len(
                config.global_mean
            ), "Error, feature dim {} is not equals to cmvn dim {}".format(
                config.filterbank_channel_count * config.channel,
                len(config.global_mean),
            )
        print("Fbank params: ", self.config)

    @classmethod
    def params(cls, config=None):
        """
    Set params.
    :param config: contains thirteen optional parameters:upper_frequency_limit(float, default=0),
    lower_frequency_limit(float, default=60.0), filterbank_channel_count(float, default=40.0),
    :return: An object of class HParams, which is a set of hyperparameters as name-value pairs.
    """

        hparams = HParams(cls=cls)

        # spectrum
        hparams.append(Spectrum.params({"output_type": 1, "is_fbank": True}))

        # fbank
        upper_frequency_limit = 0
        lower_frequency_limit = 60
        filterbank_channel_count = 40
        hparams.add_hparam("upper_frequency_limit", upper_frequency_limit)
        hparams.add_hparam("lower_frequency_limit", lower_frequency_limit)
        hparams.add_hparam("filterbank_channel_count", filterbank_channel_count)

        # delta
        delta_delta = False  # True
        order = 2
        window = 2
        hparams.add_hparam("delta_delta", delta_delta)
        hparams.add_hparam("order", order)
        hparams.add_hparam("window", window)

        if config is not None:
            hparams.parse(config, True)

        hparams.type = "Fbank"

        hparams.add_hparam("channel", 1)
        if hparams.delta_delta:
            hparams.channel = hparams.order + 1

        return hparams

    def call(self, audio_data, sample_rate):
        """
           Caculate fbank features of audio data.
           :param audio_data: the audio signal from which to compute spectrum. Should be an (1, N) tensor.
           :param sample_rate: the samplerate of the signal we working with.
           :return: A float tensor of size (num_channels, num_frames, num_frequencies) containing
                   fbank features of every frame in speech.
           """
        p = self.config

        with tf.name_scope('fbank'):

            spectrum = self.spect(audio_data, sample_rate)
            spectrum = tf.expand_dims(spectrum, 0)
            sample_rate = tf.cast(sample_rate, dtype=tf.int32)

            fbank = py_x_ops.fbank(spectrum,
                                   sample_rate,
                                   upper_frequency_limit=p.upper_frequency_limit,
                                   lower_frequency_limit=p.lower_frequency_limit,
                                   filterbank_channel_count=p.filterbank_channel_count)

            fbank = tf.squeeze(fbank, axis=0)
            shape = tf.shape(fbank)
            nframe = shape[0]
            nfbank = shape[1]
            if p.delta_delta:
                fbank = py_x_ops.delta_delta(fbank, p.order, p.window)
            if p.type == 'Fbank':
                fbank = self.cmvn(fbank)

            fbank = tf.reshape(fbank, (nframe, nfbank, p.channel))

            return fbank

    def dim(self):
        p = self.config
        return p.filterbank_channel_count

    def num_channels(self):
        p = self.config
        return p.channel
