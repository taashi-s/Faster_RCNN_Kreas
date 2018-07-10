"""
TODO : Write description
Region Proporsal Layer Module
"""

import numpy as np
import tensorflow as tf
import keras.backend as KB
from keras.layers.core import Lambda

class RegionProporsal():
    """
    TODO : Write description
    Region Proporsal Layer class

    Expecte inputs shape : [ class_logits : [B, anchor_boxes, 2]
                           , region_offsets : [B, anchor_boxes, 4]
                           ]

    """

    def __init__(self, anchors, count_limit_pre=6000, count_limit_post=2000
                 , image_shape=None, threshould=0.7, refinement_std_dev=None):
        self.__anchors = anchors
        self.__cl_pre = count_limit_pre
        self.__cl_post = count_limit_post
        self.__image_shape = image_shape
        self.__th = threshould
        self.__ref_sd = np.array([0.1, 0.1, 0.2, 0.2])
        if refinement_std_dev is not None:
            self.__ref_sd = refinement_std_dev
        self.__layer = Lambda(self.__region_proporsal
                              , output_shape=self.__region_proporsal_output_shape)


    def __call__(self):
        return self.__layer


    def __region_proporsal(self, inputs):
        scores = inputs[0][:, :, 1]
        offsets = inputs[1] * self.__ref_sd
        anchor_num = self.__anchors.shape[0]

        limit_pre = min(self.__cl_pre, anchor_num)
        top_ids = tf.nn.top_k(scores, limit_pre, sorted=True).indices

        (batch_size, _) = KB.int_shape(top_ids)
        top_posisions = self.__make_top_posisions(top_ids)
        top_scores = KB.reshape(tf.gather_nd(scores, top_posisions)
                                , [batch_size, limit_pre, 1])
        top_offsets = KB.reshape(tf.gather_nd(offsets, top_posisions)
                                 , [batch_size, limit_pre, 4])

        anchors_tile = KB.reshape(KB.tile(self.__anchors, [batch_size, 1])
                                  , [batch_size, anchor_num, 4])
        regions = KB.reshape(tf.gather_nd(anchors_tile, top_posisions)
                             , [batch_size, limit_pre, 4])

        bundle_up_regs = KB.reshape(regions, [batch_size * limit_pre, 4])
        bundle_up_offs = KB.reshape(top_offsets, [batch_size * limit_pre, 4])
        offset_region_tmp = self.__get_offset_region(bundle_up_regs, bundle_up_offs)
        if self.__image_shape is not None:
            (img_h, img_w, _) = self.__image_shape
            offset_region_tmp = KB.clip(offset_region_tmp, [0, 0, 0, 0]
                                        , [img_h, img_w, img_h, img_w])
        offset_region = KB.reshape(offset_region_tmp, [batch_size, limit_pre, 4])

        split_tile = KB.tile([1], [batch_size])
        top_data = zip(tf.split(offset_region, split_tile, axis=0)
                       , tf.split(top_scores, split_tile, axis=0))
        proposal_regions = [self.__nms(region, score) for region, score in top_data]
        return KB.stack(proposal_regions)


    def __make_top_posisions(self, top_ids):
        (batch, ids) = KB.int_shape(top_ids)
        id_base = KB.reshape(KB.arange(batch), [-1, 1])
        first_dim_ids = KB.repeat(id_base, ids)
        return KB.stack(KB.flatten(first_dim_ids), KB.flatten(top_ids), axis=1)


    def __get_offset_region(self, regions, offsets):
        reg_h = regions[:, 2] - regions[:, 0]
        reg_w = regions[:, 3] - regions[:, 1]

        pos_h = KB.exp(offsets[:, 2]) * reg_h
        pos_w = KB.exp(offsets[:, 3]) * reg_w
        center_y = offsets[:, 0] * reg_h + regions[:, 0] + 0.5 * reg_h
        center_x = offsets[:, 1] * reg_w + regions[:, 1] + 0.5 * reg_w

        min_x = center_x - 0.5 * pos_w
        min_y = center_y - 0.5 * pos_h
        max_x = center_x + 0.5 * pos_w
        max_y = center_y + 0.5 * pos_h
        return KB.transpose(KB.stack((min_y, min_x, max_y, max_x), axis=0))


    def __nms(self, regions, scores):
        if self.__image_shape is None:
            return tf.reshape(regions, [self.__cl_pre, 4])
        (img_h, img_w, _) = self.__image_shape

        regs_2d = tf.reshape(regions, [self.__cl_pre, 4])
        scos_2d = tf.reshape(scores, [self.__cl_pre])
        norm_regs = self.__normalize_regions(regs_2d, img_h, img_w)
        ids = tf.image.non_max_suppression(norm_regs, scos_2d, self.__cl_post
                                           , iou_threshold=self.__th)
        regs = tf.gather(norm_regs, ids)
        padding = tf.maximum(self.__cl_post - tf.shape(regs)[0], 0)
        padding_regs = tf.pad(regs, [(0, padding), (0, 0)])
        return padding_regs


    def __normalize_regions(self, regions, img_h, img_w):
        return regions / KB.variable([img_h, img_w, img_h, img_w])


    def __region_proporsal_output_shape(self, inputs_shape):
        shape_list = inputs_shape.as_list()
        return (shape_list[0], self.__cl_post, 4)


    def get_output_shape(self, inputs_shape):
        """
        TODO : Write description
        get_output_shape
        """
        return self.__region_proporsal_output_shape(inputs_shape)


    def get_layer(self):
        """
        TODO : Write description
        get_layer
        """
        return self.__layer
