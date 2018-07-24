"""
TODO : Write description
Region proposal Region Loss Layer Module
"""

import tensorflow as tf
from keras.layers.core import Lambda

from .utils import loss_utils as lu

class RPRegionLoss():
    """
    TODO : Write description
    Region proposal Region Loss Layer class
    """

    def __init__(self, name='rp_region_loss'):
        self.__layer = Lambda(lambda inputs: self.__region_loss(*inputs)
                              , output_shape=self.__region_loss_output_shape
                              , name=name)


    def __call__(self, inputs):
        return self.__layer(inputs)


    def __region_loss(self, cls_labels, reg_labels, preds):
        print('@@@ [RPRegLos] cls_labels : ', cls_labels)
        print('@@@ [RPRegLos] reg_labels : ', reg_labels)
        print('@@@ [RPRegLos] preds : ', preds)

        base_labels = tf.squeeze(cls_labels, -1)
        print('@@@ [RPRegLos] base_labels : ', base_labels)
        ids = tf.where(base_labels > -1)
        print('@@@ [RPRegLos] ids : ', ids)

        target_reg_labels = tf.gather_nd(reg_labels, ids)
        print('@@@ [RPRegLos] target_reg_labels : ', target_reg_labels)
        target_preds = tf.gather_nd(preds, ids)
        print('@@@ [RPRegLos] target_preds : ', target_preds)

        return lu.offset_labels_mean_loss(target_reg_labels, target_preds)


    def __region_loss_output_shape(self, _):
        return [None, 1]
