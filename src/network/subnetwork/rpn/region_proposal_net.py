"""
TODO : Write description
Region Proposal Network Module
"""

import tensorflow as tf
from keras.models import Model
from keras.engine.topology import Input
from keras.layers.core import Reshape, Activation, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization

from .layers import RegionProposal

class RegionProposalNet():
    """
    TODO : Write description
    Region Proposal Net class
    """

    def __init__(self, input_shape, anchors, input_layers=None, prev_layers=None
                 , box_by_anchor=9, image_shape=None, batch_size=5
                 , is_predict=False, trainable=True):
        self.__trainable = trainable
        self.__input_shape = input_shape

        inputs = Input(self.__input_shape)
        if input_layers is not None:
            inputs = input_layers

        prevs = inputs
        if prev_layers is not None:
            prevs = prev_layers

        #intermediate = Conv2D(256, 3, strides=3, activation="relu", padding='same'
        inter = Conv2D(256, 3, padding='same'
                              , kernel_initializer='he_uniform'
                              , trainable=self.__trainable)(prevs)

        inter_bn = BatchNormalization()(inter)
        intermediate = Activation('relu')(inter_bn)

        #cls_layer = Conv2D(2 * box_by_anchor, 1, activation="linear"
        cls_layer = Conv2D(2 * box_by_anchor, 1
                           , kernel_initializer='glorot_uniform'
                           , trainable=self.__trainable)(intermediate)
#        cls_layer = BatchNormalization()(cls_layer)
        cls_layer = Activation('linear')(cls_layer)

        # [B, h, w, box_by_anchor * 2] -> [B, anchor boxes, 2]
        cls_logits = Reshape([-1, 2])(cls_layer)
        cls_probs = Activation('softmax')(cls_logits)

        #reg_layer = Conv2D(4 * box_by_anchor, 1, activation="linear"
        reg_layer = Conv2D(4 * box_by_anchor, 1
                           , kernel_initializer='he_uniform'
                           , trainable=self.__trainable)(intermediate)
        reg_layer = BatchNormalization()(reg_layer)
        reg_layer = Activation('linear')(reg_layer)


        # [B, h, w, box_by_anchor * 4] -> [B, anchor boxes, 4]
        regions = Reshape([-1, 4])(reg_layer)

        prop_regs = RegionProposal(anchors, image_shape=image_shape, batch_size=batch_size
                                   , count_limit_pre=30 if is_predict else 60
                                   , count_limit_post=10 if is_predict else 20
                                  )([cls_probs, regions])

        outputs = ([cls_probs, regions, prop_regs])
        self.__network = outputs
        self.__model = Model(inputs=[inputs], outputs=outputs)

    def get_input_size(self):
        """
        TODO : Write description
        get_model
        """
        return self.__input_shape


    def get_network(self):
        """
        TODO : Write description
        get_model
        """
        return self.__network


    def get_model(self):
        """
        TODO : Write description
        get_model
        """
        return self.__model
