"""
TODO : Write description
Region proposal Network Module
"""

from keras.models import Model
from keras.engine.topology import Input
from keras.layers.core import Reshape, Activation
from keras.layers.convolutional import Conv2D

from layers.region_proposal import Regionproposal

class RegionproposalNet():
    """
    TODO : Write description
    RegionproposalNet class
    """

    def __init__(self, input_shape, anchors, input_layers=None, box_by_anchor=9
                 , image_shape=None, is_predict=False, trainable=True):
        self.__trainable = trainable
        self.__input_shape = input_shape

        inputs = Input(self.__input_shape)
        if input_layers is not None:
            inputs = input_layers

        intermediate = Conv2D(256, 3, strides=3, activation="relu", padding='same'
                              , trainable=self.__trainable)(inputs)

        cls_layer = Conv2D(2 * box_by_anchor, 1, activation="relu"
                           , trainable=self.__trainable)(intermediate)

        # [B, h, w, box_by_anchor * 2] -> [B, anchor boxes, 2]
        cls_logits = Reshape([-1, 2])(cls_layer)
        cls_probs = Activation('softmax')(cls_logits)

        reg_layer = Conv2D(4 * box_by_anchor, 1, activation="relu"
                           , trainable=self.__trainable)(intermediate)

        # [B, h, w, box_by_anchor * 4] -> [B, anchor boxes, 4]
        regions = Reshape([-1, 4])(reg_layer)

        prop_regs = Regionproposal(anchors, image_shape=image_shape
                                   , count_limit_post=1000 if is_predict else 2000
                                  )([cls_probs, regions])

        outputs = [cls_probs, regions, prop_regs]
        self.__network = outputs
        self.__model = Model(inputs=[inputs], outputs=[outputs])

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
