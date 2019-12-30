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

"""Builds the TensorFlow computational graph.

Tensors flowing into a single vertex are added together for all vertices
except the output, which is concatenated instead. Tensors flowing out of input
are always added.

If interior edge channels don't match, drop the extra channels (channels are
guaranteed non-decreasing). Tensors flowing out of the input as always
projected instead.
"""

import base_ops
import numpy as np
import tensorflow as tf

import math


class OneShotCell(tf.keras.layers.Layer):

    def __init__(self, spec, channels):
        super().__init__()
        self.num_vertices = np.shape(spec.matrix)[0]
        self.spec = spec
        in_degree = np.sum(spec.matrix[1:], axis=0)
        self.vertex_channels = channels // in_degree[-1]

        self.input_op = [None] * self.num_vertices
        for i in range(self.num_vertices):
            if spec.matrix[0, i]:
                self.input_op[i] = base_ops.ConvBnRelu(1, self.vertex_channels)
        self.op = [None] * self.num_vertices
        for i in range(1, self.num_vertices - 1):
            self.op[i] = base_ops.OP_MAP[spec.ops[i]](self.vertex_channels)

    def call(self, inputs):
        tensors = [inputs]
        final_concat_in = []
        for t in range(1, self.num_vertices - 1):
            add_in = [tensors[src] for src in range(1, t) if self.spec.matrix[src, t]]
            if self.spec.matrix[0, t]:
                add_in.append(self.input_op[t](tensors[0]))
            vertex_value = self.op[t](sum(add_in))
            tensors.append(vertex_value)
            if self.spec.matrix[t, self.num_vertices - 1]:
                final_concat_in.append(vertex_value)
        outputs = tf.keras.layers.concatenate(final_concat_in, -1)
        if self.spec.matrix[0, self.num_vertices - 1]:
            outputs += self.input_op[self.num_vertices - 1](inputs)
        return outputs


class Network(tf.keras.Model):
    def __init__(self, spec, config):
        super().__init__()

        layers = []

        # initial stem convolution
        stem_conv = base_ops.ConvBnRelu(3, config['stem_filter_size'])
        layers.append(stem_conv)

        channels = config['stem_filter_size']
        for stack_num in range(config['num_stacks']):
            if stack_num > 0:
                downsample = tf.keras.layers.MaxPool2D(pool_size=2)
                layers.append(downsample)
                channels *= 2
            for _ in range(config['num_modules_per_stack']):
                cell = OneShotCell(spec, channels)
                layers.append(cell)

        self.features = tf.keras.Sequential(layers)
        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.classifier = tf.keras.layers.Dense(config['num_labels'], activation="softmax")

    def call(self, inputs):
        out = self.features(inputs)
        out = self.gap(out)
        out = self.classifier(out)
        return out
