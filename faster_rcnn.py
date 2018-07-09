"""
TODO : Write description
Faster R-CNN Module
"""

from enum import Enum
import tensorflow as tf
from keras.models import Model
from keras.engine.topology import Input

from ResNet.resnet import ResNet
from region_proporsal_net import RegionProporsalNet

class TrainTarget(Enum):
    """
    TODO : Write description
    TrainTarget Enum
    """
    BACKBONE = 1
    RPN = 2
    HEAD = 3

class FasterRCNN():
    """
    TODO : Write description
    ResNet class
    """

    def __init__(self, input_shape, train_taegets=None):
        self.__input_shape = input_shape

        if train_taegets is None:
            train_taegets = []
        train_backbone = TrainTarget.BACKBONE in train_taegets
        train_rpn = TrainTarget.RPN in train_taegets
        train_head = TrainTarget.HEAD in train_taegets

        inputs = Input(self.__input_shape)

        resnet = ResNet(inputs.get_shape(), input_layers=inputs
                        , trainable=train_backbone).get_residual_network()

        rpn = RegionProporsalNet(resnet.get_shape(), input_layers=resnet
                                 , trainable=train_rpn).get_network()

        lossees_list = []
        if train_rpn:
            # calculate rpn loss, and append to lossees_list
            lossees_list += []

        if train_head:
            # add head layer
            #__head_net()

            # calculate rpn loss, and append to lossees_list
            lossees_list += []

        outputs = lossees_list
        self.__network = outputs
        self.__model = Model(inputs=[inputs], outputs=[outputs])

        for lossees in lossees_list:
            self.__model.add_loss(tf.reduce_mean(lossees, keep_dims=True))


    def __head_net(self, inputs):
        # RoI Pooling
        # outputs = (N, box, reg_w, reg_h, ch)

        # Conv2 * 2 by box
        # Use TimeDistributed()

        # Classification by box
        # Use TimeDistributed()

        # Region Detection by box
        # Use TimeDistributed()

        return inputs


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
