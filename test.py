import math
import tensorflow as tf

from cifar import CIFARInput
from config import build_config
from model_spec import ModelSpec
from model_builder import Network


def add_l2_regularizer(model, weight_decay):
    regularizer = tf.keras.regularizers.l2(weight_decay)
    for layer in model.layers:
        for attr in ['kernel_regularizer', 'bias_regularizer', 'beta_regularizer', 'gamma_regularizer']:
            setattr(layer, attr, regularizer)


if __name__ == '__main__':
    # model_spec = ModelSpec([[0, 1, 1, 0, 1, 0, 0],
    #                         [0, 0, 0, 0, 0, 1, 0],
    #                         [0, 0, 0, 1, 0, 0, 0],
    #                         [0, 0, 0, 0, 1, 0, 0],
    #                         [0, 0, 0, 0, 0, 1, 1],
    #                         [0, 0, 0, 0, 0, 0, 1],
    #                         [0, 0, 0, 0, 0, 0, 0]],
    #                        ['input', 'maxpool3x3', 'maxpool3x3', 'conv3x3-bn-relu', 'conv1x1-bn-relu', 'conv3x3-bn-relu', 'output']
    #                        )
    model_spec = ModelSpec([[0, 1, 1, 1, 1, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0]],
                           ['input', 'conv3x3-bn-relu', 'conv1x1-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3', 'maxpool3x3', 'output'])
    config = build_config()
    # network = tf.keras.applications.ResNet50(include_top=True, weights=None, input_shape=(32, 32, 3), classes=10)
    network = Network(model_spec, config)
    add_l2_regularizer(network, config['weight_decay'])

    batch_size = config['batch_size']
    dataset_train = CIFARInput('train', config)
    dataset_valid = CIFARInput('valid', config)
    steps_per_epoch = int(math.ceil(dataset_train.num_images / batch_size))
    validation_steps_per_epoch = int(math.ceil(dataset_valid.num_images / batch_size))

    epochs = config['train_epochs']
    learning_rate = tf.keras.experimental.CosineDecay(config['learning_rate'], epochs * steps_per_epoch)
    # optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, momentum=config['momentum'], epsilon=1)

    cosine_decay = lambda step: tf.compat.v1.train.cosine_decay(config['learning_rate'], step, epochs * steps_per_epoch)
    optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=cosine_decay, momentum=config['momentum'], epsilon=1.)
    cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    network.compile(optimizer=optimizer, loss=cce, metrics=['accuracy'])

    network.fit(dataset_train.input_fn(batch_size), steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=1,
                validation_steps=validation_steps_per_epoch, validation_freq=1, validation_data=dataset_valid.input_fn(batch_size))
