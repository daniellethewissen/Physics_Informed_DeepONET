import numpy as np
import tensorflow as tf

class Loss(tf.keras.losses.Loss):
    def __init__(self, batch_size=None, batch_size_val=None):

        self.batch_size = batch_size
        self.batch_size_val = batch_size_val

        

    def call(self, y_true_1,  y_true_2,  y_true_3,  y_true_4,  y_true_5, y_pred_1, y_pred_2, y_pred_3,y_pred_4, y_pred_5):       
        loss = tf.reduce_mean(tf.square(y_true_1 - y_pred_1)+tf.square(y_true_2 - y_pred_2)+tf.square(y_true_3 - y_pred_3)+tf.square(y_true_4 - y_pred_4)+tf.square(y_true_5 - y_pred_5)) 
        return loss