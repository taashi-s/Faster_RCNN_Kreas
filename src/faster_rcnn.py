"""
TODO : Write description
Faster R-CNN Module
"""

from enum import Enum
import tensorflow as tf
from keras.models import Model
from keras.engine.topology import Input
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, Flatten, Activation, Reshape
from keras.layers.normalization import BatchNormalization

from subnetwork.resnet.resnet import ResNet
from subnetwork.rpn.region_proporsal_net import RegionProporsalNet
from layers.rp_class_loss import RPClassLoss
from layers.rp_region_loss import RPRegionLoss
from layers.detection_target_region import DetectionTargetRegion
from layers.class_loss import ClassLoss
from layers.region_loss import RegionLoss
from layers.roi_pooling import RoIPooling

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

    def __init__(self, input_shape, class_num, train_taegets=None):
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
            head = self.__head_net(dtr, class_num)

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


    def __head_net(self, inputs, class_num):
        roi_pool = RoIPooling()(inputs)
        flt = TimeDistributed(Flatten())(roi_pool)

        fc1 = TimeDistributed(Dense(2048))(flt)
        norm1 = TimeDistributed(BatchNormalization())(fc1)
        act1 = TimeDistributed(Activation('relu'))(norm1)
        fc2 = TimeDistributed(Dense(2048))(act1)
        norm2 = TimeDistributed(BatchNormalization())(fc2)
        act2 = TimeDistributed(Activation('relu'))(norm2)
        outputs = act2

        cls_logits = TimeDistributed(Dense(class_num))(outputs)
        cls_probs = Activation('softmax')(cls_logits)

        reg_tmp = TimeDistributed(Dense(class_num * 4))(outputs)
        regions = Reshape([-1, class_num, 4])(reg_tmp)

        return cls_logits, cls_probs, regions


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
