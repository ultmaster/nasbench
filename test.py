import tensorflow as tf

from cifar import CIFARInput
from config import build_config
from model_spec import ModelSpec
from model_builder import Network


if __name__ == "__main__":
    model_spec = ModelSpec([[0, 1, 0, 1, 0, 0, 0],
                            [0, 0, 1, 0, 0, 1, 1],
                            [0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0]],
                           ['input', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3',
                            'maxpool3x3', 'output']
                           )
    config = build_config()
    network = Network(model_spec, config)
    steps_per_epoch = 157
    epochs = config['train_epochs']
    lr_decayed_fn = tf.keras.experimental.CosineDecay(config['learning_rate'], epochs * steps_per_epoch)
    # optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_decayed_fn, momentum=config['momentum'], epsilon=1)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_decayed_fn, momentum=config['momentum'])
    network.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    dataset = CIFARInput('train', config)
    network.fit(dataset.input_fn(config), steps_per_epoch=steps_per_epoch, epochs=epochs)
