"""
TODO : Write description
Faster R-CNN Module
"""

from enum import Enum
import tensorflow as tf
from keras.models import Model
from keras.engine.topology import Input

from subnetwork.resnet.resnet import ResNet
from subnetwork.rpn.region_proporsal_net import RegionProporsalNet
from layers.rp_class_loss import RPClassLoss
from layers.rp_region_loss import RPRegionLoss
from layers.detection_target_region import DetectionTargetRegion
from layers.class_loss import ClassLoss
from layers.region_loss import RegionLoss

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

        inputs = []
        outputs = []

        inputs_images = Input(shape=self.__input_shape)
        inputs += [inputs_images]

        resnet = ResNet(inputs_images.get_shape(), input_layers=inputs_images
                        , trainable=train_backbone).get_residual_network()

        rpn = RegionProporsalNet(resnet.get_shape(), input_layers=resnet
                                 , trainable=train_rpn).get_network()

        if train_rpn:
            inputs_rp_cls = Input(shape=[None, 1], dtype='int32')
            inputs_rp_reg = Input(shape=[None, 4], dtype='float32')
            inputs += [inputs_rp_cls, inputs_rp_reg]

            rp_cls_losses = RPClassLoss()(rpn)
            rp_reg_losses = RPRegionLoss()(rpn)
            outputs += [rp_cls_losses, rp_reg_losses]

        if train_head:
            dtr = DetectionTargetRegion()(rpn)
            head = self.__head_net(dtr)

            inputs_cls = Input(shape=[None, 1], dtype='int32')
            inputs_reg = Input(shape=[None, 4], dtype='float32')
            inputs += [inputs_cls, inputs_reg]

            cls_losses = ClassLoss()(head)
            reg_losses = RegionLoss()(head)
            outputs += [cls_losses, reg_losses]

        self.__network = outputs
        self.__model = Model(inputs=[inputs], outputs=[outputs])

        for output in outputs:
            self.__model.add_loss(tf.reduce_mean(output, keep_dims=True))


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
