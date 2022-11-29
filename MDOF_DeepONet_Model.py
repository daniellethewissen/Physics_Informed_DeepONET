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

class UnstackedMultiOutputDeepONet(tf.keras.Model):

    def __init__(
        self,
        layer_sizes_branch,
        layer_sizes_trunk,
        activation_branch,
        activation_trunk,
        kernel_initializer,
    ):
        super().__init__()
        self.activation_trunk = tf.keras.activations.get(activation_trunk[-1])
        # Fully connected network
        self.branch_1 = FNN(layer_sizes_branch, activation_branch, kernel_initializer)
        self.branch_2 = FNN(layer_sizes_branch, activation_branch, kernel_initializer)
        self.branch_3 = FNN(layer_sizes_branch, activation_branch, kernel_initializer)
        self.branch_4 = FNN(layer_sizes_branch, activation_branch, kernel_initializer)
        self.branch_5 = FNN(layer_sizes_branch, activation_branch, kernel_initializer)
        self.trunk_1 = FNN(layer_sizes_trunk, activation_trunk, kernel_initializer)
        self.trunk_2 = FNN(layer_sizes_trunk, activation_trunk, kernel_initializer)
        self.trunk_3 = FNN(layer_sizes_trunk, activation_trunk, kernel_initializer)
        self.trunk_4 = FNN(layer_sizes_trunk, activation_trunk, kernel_initializer)
        self.trunk_5 = FNN(layer_sizes_trunk, activation_trunk, kernel_initializer)
        self.b_1 = tf.Variable(tf.zeros(1))
        self.b_2 = tf.Variable(tf.zeros(1))
        self.b_3 = tf.Variable(tf.zeros(1))
        self.b_4 = tf.Variable(tf.zeros(1))
        self.b_5 = tf.Variable(tf.zeros(1))

    def call(self, branch_inputs, trunk_inputs, training=False):
        # Branch net to encode the input function
        x_func_1 = self.branch_1(branch_inputs)
        x_func_2 = self.branch_2(branch_inputs)
        x_func_3 = self.branch_3(branch_inputs)
        x_func_4 = self.branch_4(branch_inputs)
        x_func_5 = self.branch_5(branch_inputs)
        
        # Trunk net to encode the domain of the output function
        x_loc_1 = self.trunk_1(trunk_inputs)
        x_loc_2 = self.trunk_2(trunk_inputs)
        x_loc_3 = self.trunk_3(trunk_inputs)
        x_loc_4 = self.trunk_4(trunk_inputs)
        x_loc_5 = self.trunk_5(trunk_inputs)


        # Dot product
        if x_func_1.shape[-1] != x_loc_1.shape[-1]:
            raise AssertionError(
                "Output sizes of branch net and trunk net do not match."
            )
        
        x_1 = tf.multiply(x_func_1, x_loc_1)
        x_1 = tf.reduce_sum(x_1, axis=1)
        x_1 = tf.expand_dims(x_1, axis=1)

        x_2 = tf.multiply(x_func_2, x_loc_2)
        x_2 = tf.reduce_sum(x_2, axis=1)
        x_2 = tf.expand_dims(x_2, axis=1)

        x_3 = tf.multiply(x_func_3, x_loc_3)
        x_3 = tf.reduce_sum(x_3, axis=1)
        x_3 = tf.expand_dims(x_3, axis=1)

        x_4 = tf.multiply(x_func_4, x_loc_4)
        x_4 = tf.reduce_sum(x_4, axis=1)
        x_4 = tf.expand_dims(x_4, axis=1)

        x_5 = tf.multiply(x_func_5, x_loc_5)
        x_5 = tf.reduce_sum(x_5, axis=1)
        x_5 = tf.expand_dims(x_5, axis=1)
        
        # Add bias
        x_1 += self.b_1
        x_2 += self.b_2
        x_3 += self.b_3
        x_4 += self.b_4
        x_5 += self.b_5

        # x_1 = tf.keras.activations.sigmoid(x_1)
        # x_2 = tf.keras.activations.sigmoid(x_2)
        # x_3 = tf.keras.activations.sigmoid(x_3)

        # x_1 = tf.keras.activations.relu(x_1)
        # x_2 = tf.keras.activations.relu(x_2)
        # x_3 = tf.keras.activations.relu(x_3)

        
        return x_1, x_2, x_3, x_4, x_5

