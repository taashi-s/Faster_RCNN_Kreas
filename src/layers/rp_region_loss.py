"""
TODO : Write description
Region proposal Region Loss Layer Module
"""

import tensorflow as tf
import keras.backend as KB
from keras.layers.core import Lambda

class RPRegionLoss():
    """
    TODO : Write description
    Region proposal Region Loss Layer class
    """

    def __init__(self):
        self.layer = Lambda(lambda inputs: self.__region_loss(*inputs)
                            , output_shape=self.__region_loss_output_shape)


    def __call__(self):
        return self.layer


    def __region_loss(self, cls_labels, reg_labels, preds):
        base_labels = tf.squeeze(cls_labels, -1)
        ids = tf.where(base_labels > -1)

        target_reg_labels = tf.gather_nd(reg_labels, ids)
        target_preds = tf.gather_nd(preds, ids)

        return self.__reg_labels_mean_loss(target_reg_labels, target_preds)


    def __reg_labels_mean_loss(self, labels, preds):
        loss = self.__smooth(labels, preds)
        return KB.switch(tf.size(loss) > 0, KB.mean(loss), KB.constant(0.0))


    def __smooth(self, labels, preds):
        diff = KB.abs(labels - preds)
        less = KB.cast(KB.less(diff, 1.0), "float32")
        return (less * 0.5 * diff**2) + (1 - less) * (diff - 0.5)


    def __region_loss_output_shape(self, _):
        return [1]
