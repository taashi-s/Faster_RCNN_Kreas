"""
TODO : Write description
Region proposal Class Loss Layer Module
"""

import tensorflow as tf
from keras.layers.core import Lambda

import utils.loss_utils as lu

class RPClassLoss():
    """
    TODO : Write description
    Region proposal Class Loss Layer class
    """

    def __init__(self):
        self.layer = Lambda(lambda inputs: self.__class_loss(*inputs)
                            , output_shape=self.__class_loss_output_shape)


    def __call__(self):
        return self.layer


    def __class_loss(self, labels, preds):
        base_labels = tf.squeeze(labels, -1)
        ids = tf.where(base_labels > -1)

        target_labels = tf.gather_nd(base_labels, ids)
        target_preds = tf.gather_nd(preds, ids)

        return lu.class_labels_mean_loss(target_labels, target_preds)


    def __class_loss_output_shape(self, _):
        return [1]
