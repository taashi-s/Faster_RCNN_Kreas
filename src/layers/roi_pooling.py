"""
TODO : Write description
RoI Pooling Layer Module
"""

import tensorflow as tf
import keras.backend as KB
import keras.utils.conv_utils as KCUtils
from keras.layers.core import Lambda


class RoIPooling():
    """
    TODO : Write description
    RoI Pooling Layer class

    TODO : this processing is like the RoI Align Layer.

    this processing :
        fmaps crop, and use crop_and_resize(binary interpolation)

    actual RoI Pooling processing :
        Rounding the coordinates of the regions.
        After that, separate the region into pooling size
        , then max(or avarage) pooling in that each separated region.

    actual RoI Align processing :
        Separate the region into pooling size.
        (without, rounding the coordinates of the regions),
        Calculate four corners of that each separate region
        from each four neighborhoods by bilinear interpolation.
        Finally, max(or avarage) pooling that four corners.

    """

    def __init__(self, batch_size=5, pooling=(7, 7), image_shape=None):
        (pooling_h, pooling_w) = KCUtils.normalize_tuple(pooling, 2, 'pooling')
        self.__pooling_h = pooling_h
        self.__pooling_w = pooling_w
        self.__image_shape = image_shape
        self.__batch_size = batch_size
        self.__layer = Lambda(lambda inputs: self.__roi_pooling(*inputs)
                              , output_shape=self.__roi_pooling_output_shape)


    def __call__(self, inputs):
        return self.__layer(inputs)


    def __roi_pooling(self, fmaps, regions):
        reg_num = regions.get_shape()[1]

        flat_regs = KB.concatenate(tf.unstack(regions), axis=0)
        img_ids = KB.arange(self.__batch_size)
        target_img_ids = KB.flatten(KB.repeat(KB.reshape(img_ids, [-1, 1]), reg_num))
        pooling_size = [self.__pooling_h, self.__pooling_w]

        pooling_fmaps = tf.image.crop_and_resize(fmaps, flat_regs, target_img_ids, pooling_size)
        #output_shape = (self.__batch_size, reg_num, self.__pooling_h, self.__pooling_w, -1)
        output_shape = (-1, reg_num, self.__pooling_h, self.__pooling_w, pooling_fmaps.get_shape()[3])
        return KB.reshape(pooling_fmaps, output_shape)


    def __roi_pooling_output_shape(self, inputs_shape):
        return [None, inputs_shape[0][1], self.__pooling_h, self.__pooling_w, inputs_shape[0][3]]
