"""
TODO : Write description
Region proposal Class Loss Layer Module
"""

import tensorflow as tf
import keras.backend as KB
from keras.layers.core import Lambda

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

        return self.__cls_labels_mean_loss(target_labels, target_preds)


    def __cls_labels_mean_loss(self, labels, preds):
        loss = KB.sparse_categorical_crossentropy(labels, preds)
        return KB.switch(tf.size(loss) > 0, KB.mean(loss), KB.constant(0.0))


    def __class_loss_output_shape(self, _):
        return [1]
