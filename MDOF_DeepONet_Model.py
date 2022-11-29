# Import Libaries ----------------------------
import numpy as np
import tensorflow as tf

class FNN(tf.keras.Model):
    def __init__(
        self,
        layer_sizes,
        activation,
        kernel_initializer,
        #dropout_rate=0,
    ):
        super().__init__()
        #self.dropout_rate = dropout_rate

        self.denses = []
        if kernel_initializer == "Glorot Normal":
            initializer = tf.keras.initializers.GlorotNormal()
        elif kernel_initializer == "He Normal":
            initializer = tf.keras.initializers.HeNormal()
        else:
            initializer = tf.keras.initializers.RandomNormal()

        for i in range(len(layer_sizes)):
            self.denses.append(
                tf.keras.layers.Dense(
                    layer_sizes[i],
                    activation=activation[i],
                    kernel_initializer=initializer
                )
            )
            self.denses.append(
                tf.keras.layers.BatchNormalization()
                )

            # if self.dropout_rate > 0:
            #  self.denses.append(tf.keras.layers.Dropout(rate=self.dropout_rate))

    def call(self, inputs, training=False):
        y = inputs
        for layer in self.denses:
            y = layer(y, training=training)
        return y


class DeepONet(tf.keras.Model):

    def __init__(
        self,
        layer_sizes_branch,
        layer_sizes_trunk,
        activation_branch,
        activation_trunk,
        kernel_initializer
    ):
        super().__init__()
        self.activation_trunk = tf.keras.activations.get(activation_trunk[-1])
        # Fully connected network
        self.branch = FNN(layer_sizes_branch, activation_branch, kernel_initializer)
        self.trunk = FNN(layer_sizes_trunk, activation_trunk, kernel_initializer)
        self.b = tf.Variable(tf.zeros(1))

    def call(self, branch_inputs, trunk_inputs, training=False):
        # Branch net to encode the input function
        x_func = self.branch(branch_inputs)
        # Trunk net to encode the domain of the output function
        # x_loc = self.activation_trunk(self.trunk(trunk_inputs))
        x_loc = self.trunk(trunk_inputs)
        # Dot product
        if x_func.shape[-1] != x_loc.shape[-1]:
            raise AssertionError(
                "Output sizes of branch net and trunk net do not match."
            )
        # x = tf.einsum("bi,ni->bn", x_func, x_loc)
        x = tf.multiply(x_func[:, 0:], x_loc[:, 0:])
        # x = tf.reduce_sum(x, axis=1)
        x = tf.reduce_sum(x, axis=1, keepdims=True)
        # x = tf.expand_dims(x, axis=1)
        # Add bias
        # x += self.b

        return x

