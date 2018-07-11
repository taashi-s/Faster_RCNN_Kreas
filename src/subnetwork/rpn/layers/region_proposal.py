"""
TODO : Write description
Region proposal Layer Module
"""

import numpy as np
import tensorflow as tf
import keras.backend as KB
from keras.layers.core import Lambda

from utils.regions_utils import RegionsUtils
from utils.non_maximal_suppression import NMS

class Regionproposal():
    """
    TODO : Write description
    Region proposal Layer class

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
        self.__layer = Lambda(self.__region_proposal
                              , output_shape=self.__region_proposal_output_shape)


    def __call__(self):
        return self.__layer


    def __region_proposal(self, inputs):
        scores = inputs[0][:, :, 1]
        offsets = inputs[1] * self.__ref_sd

        anchor_num = self.__anchors.shape[0]
        limit_pre = min(self.__cl_pre, anchor_num)
        top_ids = tf.nn.top_k(scores, limit_pre, sorted=True).indices
        (batch_size, _) = KB.int_shape(top_ids)
        top_poss = self.__make_top_posisions(top_ids)
        top_scores = self.__get_top_scores(scores, top_poss, batch_size, limit_pre)
        top_regions = self.__get_top_regions(offsets, top_poss, batch_size, limit_pre, anchor_num)

        split_tile = KB.tile([1], [batch_size])
        top_zip = zip(tf.split(top_regions, split_tile, axis=0)
                      , tf.split(top_scores, split_tile, axis=0))
        proposal_regions = [self.__nms(rs, ss) for rs, ss in top_zip]
        return KB.stack(proposal_regions)


    def __get_top_scores(self, scores, top_posisions, batch_size, limit):
        return KB.reshape(tf.gather_nd(scores, top_posisions), [batch_size, limit, 1])


    def __get_top_regions(self, offsets, top_posisions, batch_size, limit, anchor_num):
        top_offsets = KB.reshape(tf.gather_nd(offsets, top_posisions)
                                 , [batch_size, limit, 4])
        anchors_tile = KB.reshape(KB.tile(self.__anchors, [batch_size, 1])
                                  , [batch_size, anchor_num, 4])
        regions = KB.reshape(tf.gather_nd(anchors_tile, top_posisions)
                             , [batch_size, limit, 4])

        bundle_up_regs = KB.reshape(regions, [batch_size * limit, 4])
        bundle_up_offs = KB.reshape(top_offsets, [batch_size * limit, 4])
        offset_region_tmp = RegionsUtils(bundle_up_regs).offset(bundle_up_offs)
        if self.__image_shape is not None:
            (img_h, img_w, _) = self.__image_shape
            offset_region_tmp = KB.clip(offset_region_tmp, [0, 0, 0, 0]
                                        , [img_h, img_w, img_h, img_w])
        return KB.reshape(offset_region_tmp, [batch_size, limit, 4])


    def __make_top_posisions(self, top_ids):
        (batch, ids) = KB.int_shape(top_ids)
        id_base = KB.reshape(KB.arange(batch), [-1, 1])
        first_dim_ids = KB.repeat(id_base, ids)
        return KB.stack(KB.flatten(first_dim_ids), KB.flatten(top_ids), axis=1)


    def __nms(self, regions, scores):
        regs_2d = tf.reshape(regions, [self.__cl_pre, 4])
        scos_2d = tf.reshape(scores, [self.__cl_pre])
        regs = regs_2d
        if self.__image_shape is not None:
            (img_h, img_w, _) = self.__image_shape
            regs = RegionsUtils(regs_2d).normalize(img_h, img_w)
        return NMS(self.__cl_post, self.__th)(regs, scos_2d)


    def __region_proposal_output_shape(self, inputs_shape):
        shape_list = inputs_shape.as_list()
        return (shape_list[0], self.__cl_post, 4)


    def get_output_shape(self, inputs_shape):
        """
        TODO : Write description
        get_output_shape
        """
        return self.__region_proposal_output_shape(inputs_shape)


    def get_layer(self):
        """
        TODO : Write description
        get_layer
        """
        return self.__layer
