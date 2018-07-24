"""
TODO : Write description
Region Loss Layer Module
"""

import tensorflow as tf
import keras.backend as KB
from keras.layers.core import Lambda

from .utils import loss_utils as lu

class RegionLoss():
    """
    TODO : Write description
    Region Loss Layer class
    """

    def __init__(self, name='region_loss'):
        self.__layer = Lambda(lambda inputs: self.__region_loss(*inputs)
                              , output_shape=self.__region_loss_output_shape
                              , name=name)


    def __call__(self, inputs):
        return self.__layer(inputs)


    def __region_loss(self, cls_labels, ofs_labels, preds):
        positive_ids = tf.where(cls_labels > 0)
        target_batch_ids = KB.cast(positive_ids[:, 0], 'int32')
        target_region_ids = KB.cast(positive_ids[:, 1], 'int32')
        target_class_ids = KB.cast(tf.gather_nd(cls_labels, positive_ids), 'int32')
        target_pred_ids = KB.stack((target_batch_ids, target_region_ids, target_class_ids), axis=1)

        target_label = tf.gather_nd(ofs_labels, positive_ids)
        target_pred = tf.gather_nd(preds, target_pred_ids)
        return lu.offset_labels_mean_loss(target_label, target_pred)


    def __region_loss_output_shape(self, _):
        return [None, 1]
