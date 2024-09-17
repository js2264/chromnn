import tensorflow as tf
from tensorflow.keras import layers, models


class ChromNNModel:
    def __init__(self, winsize, output_size=1):
        self.winsize = winsize
        self.output_size = output_size

    def build_model(self):
        kernel_init = tf.keras.initializers.VarianceScaling()
        inputs = layers.Input(shape=(self.winsize, 4))

        # Convolutional blocks
        x = layers.Conv1D(
            128, 15, padding="same", activation="relu", kernel_initializer=kernel_init
        )(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)

        for dilation_rate in [1, 2, 4, 8, 16]:
            x = self._add_conv_block(x, 256, dilation_rate, kernel_init)

        # Global average pooling
        x = layers.GlobalMaxPooling1D()(x)

        # Fully connected layer
        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dropout(0.3)(x)

        # Output layer
        output_layer = layers.Dense(self.output_size, activation="linear")(x)

        model = models.Model(inputs, output_layer)
        return model

    def _add_conv_block(self, x, filters, dilation_rate, kernel_init):
        x = layers.Conv1D(
            filters,
            5,
            padding="same",
            activation="relu",
            dilation_rate=dilation_rate,
            kernel_initializer=kernel_init,
        )(x)
        x = layers.BatchNormalization()(x)
        return x
