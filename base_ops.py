# Copyright 2019 The Google Research Authors.
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

"""Base operations used by the modules in this search space."""

import abc

import tensorflow as tf

# Currently, only channels_last is well supported.
VALID_DATA_FORMATS = frozenset(['channels_last', 'channels_first'])
MIN_FILTERS = 8
BN_MOMENTUM = 0.997
BN_EPSILON = 1e-5


class ConvBnRelu(layers.Layer):
    def __init__(self, conv_size, conv_filters):
        self.conv = tf.keras.layers.Conv2d(conv_filters, conv_size, use_bias=False,
                                           kernel_initializer=tf.keras.initializers.VarianceScaling())
        self.bn = tf.keras.layers.BatchNormalization(momentum=BN_EPSILON, epsilon=BN_EPSILON)

    def call(self, inputs):
        out = self.conv(inputs)
        out = self.bn(out)
        out = tf.keras.activations.relu(out)
        return out


class Conv3x3BnRelu(ConvBnRelu):
    def __init__(self, conv_filters):
        super().__init__(3, conv_filters)


class Conv1x1BnRelu(ConvBnRelu):
    def __init__(self, conv_filters):
        super().__init__(1, conv_filters):


class MaxPool3x3(layers.Layer):
    def __init__(self, conv_filters):
        self.pool = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=1, padding="same")

    def call(self, inputs):
        return self.pool(inputs)


# Commas should not be used in op names
OP_MAP = {
    'conv3x3-bn-relu': Conv3x3BnRelu,
    'conv1x1-bn-relu': Conv1x1BnRelu,
    'maxpool3x3': MaxPool3x3
}
