import numpy as np
import tensorflow as tf

class Loss(tf.keras.losses.Loss):
    def __init__(self, batch_size=None, batch_size_val=None):

        self.batch_size = batch_size
        self.batch_size_val = batch_size_val

        

    def call(self, y_true, y_pred):       
        loss = tf.reduce_mean(tf.square(y_true - y_pred)) 
        return loss