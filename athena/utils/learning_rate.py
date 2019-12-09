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
# pylint: disable=too-few-public-methods, no-member, too-many-arguments
""" learning rate """
import tensorflow as tf
from ..utils.hparam import HParams


class WarmUpLearningSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """ WarmUp Learning rate schedule for Adam

    Used as :
        optimizer = tf.keras.optimizers.Adam(learning_rate = WarmUpLearningSchedule(512),
            beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    Args :
        model_dim is the something related to total model parameters
        warmup_steps is the highest learning rate iters
    Returns:
        return the learning rate
    Idea from the paper: Attention Is All You Need
    """

    def __init__(self, model_dim=512, warmup_steps=4000, k=1.0):
        super().__init__()

        self.model_dim = tf.cast(model_dim, tf.float32)
        self.warmup_steps = warmup_steps
        self.k = k

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return self.k * tf.math.rsqrt(self.model_dim) * tf.math.minimum(arg1, arg2)


class WarmUpAdam(tf.keras.optimizers.Adam):
    """WarmUpAdam Implementation """

    def __init__(self, config=None, beta_1=0.9, beta_2=0.999, epsilon=1e-7,
                 amsgrad=False, name="WarmUpAdam", **kwargs):
        self.hparams = HParams(cls=self.__class__)
        self.hparams.add_hparam("d_model", 512)
        self.hparams.add_hparam("warmup_steps", 4000)
        self.hparams.add_hparam("k", 1.0)
        if config is not None:
            self.hparams.override_from_dict(config)
        d_model = self.hparams.d_model
        warmup_steps = self.hparams.warmup_steps
        k = self.hparams.k
        super().__init__(
            learning_rate=WarmUpLearningSchedule(d_model, warmup_steps, k),
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            amsgrad=amsgrad,
            name=name,
        )


class ExponentialDecayLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """ ExponentialDecayLearningRateSchedule

    Used as :
        optimizer = tf.keras.optimizers.Adam(
        learning_rate = ExponentialDecayLearningRate(0.01, 100))
    Args :
        initial_lr, decay_steps
    Returns:
        initial_lr * (0.5 ** (step // decay_steps))
    """

    def __init__(self, initial_lr=0.005, decay_steps=10000):
        super().__init__()
        self.initial_lr = initial_lr
        self.decay_steps = tf.cast(decay_steps, tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        factor = tf.cast(0.5 ** (step // self.decay_steps), tf.float32)
        return self.initial_lr * factor


class ExponentialDecayAdam(tf.keras.optimizers.Adam):
    """WarmUpAdam Implementation """

    def __init__(self, config=None, beta_1=0.9, beta_2=0.999, epsilon=1e-7,
                 amsgrad=False, name="WarmUpAdam", **kwargs):
        self.hparams = HParams(cls=self.__class__)
        self.hparams.add_hparam("initial_lr", 0.005)
        self.hparams.add_hparam("decay_steps", 10000)
        if config is not None:
            self.hparams.override_from_dict(config)
        initial_lr = self.hparams.initial_lr
        decay_steps = self.hparams.decay_steps
        super().__init__(
            learning_rate=ExponentialDecayLearningRateSchedule(initial_lr, decay_steps),
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            amsgrad=amsgrad,
            name=name,
        )
