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
from keras.optimizers import SGD
from keras.utils import plot_model

from subnetwork.resnet.resnet import ResNet
from subnetwork.rpn.region_proposal_net import RegionproposalNet
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

    def __init__(self, input_shape, class_num, anchors
                 , batch_size=5, is_predict=False, train_taegets=None):
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
                        , trainable=train_backbone
                       ).get_residual_network()

        rpn = RegionproposalNet(resnet.get_shape(), anchors, input_layers=inputs_images
                                , image_shape=self.__input_shape, prev_layers=resnet
                                , batch_size=batch_size, is_predict=is_predict
                                , trainable=train_rpn
                               ).get_network()
        rpn_cls_probs, rpn_regions, rpn_prop_regs = rpn

        if train_rpn and not is_predict:
            inputs_rp_cls = Input(shape=[None, 1], dtype='int32')
            inputs_rp_reg = Input(shape=[None, 4], dtype='float32')
            inputs += [inputs_rp_cls, inputs_rp_reg]

            rp_cls_losses = RPClassLoss()([inputs_rp_cls, rpn_cls_probs])
            rp_reg_losses = RPRegionLoss()([inputs_rp_cls, inputs_rp_reg, rpn_regions])
            outputs += [rp_cls_losses, rp_reg_losses]

        if train_head and not is_predict:
            inputs_cls = Input(shape=[None, 1], dtype='int32')
            inputs_reg = Input(shape=[None, 4], dtype='float32')
            inputs += [inputs_cls, inputs_reg]

            dtr = DetectionTargetRegion(positive_threshold=0.5, positive_ratio=0.33
                                        , image_shape=self.__input_shape, batch_size=batch_size
                                        , exclusion_threshold=0.1, count_per_batch=64
                                       )([inputs_cls, inputs_reg, rpn_prop_regs])
            dtr_cls_labels, dtr_offsets_labels, dtr_regions = dtr
            clsses, offsets = self.__head_net(resnet, dtr_regions, class_num, batch_size=batch_size)

            cls_losses = ClassLoss()([dtr_cls_labels, clsses])
            reg_losses = RegionLoss()([dtr_cls_labels, dtr_offsets_labels, offsets])
            outputs += [cls_losses, reg_losses]

        if is_predict:
            clsses, offsets = self.__head_net(resnet, rpn_prop_regs, class_num
                                              , batch_size=batch_size)
            outputs = [rpn_prop_regs, clsses, offsets]

        self.__network = outputs
        self.__model = Model(inputs=inputs, outputs=outputs)

        for output in outputs:
            self.__model.add_loss(tf.reduce_mean(output))


    def __head_net(self, fmaps, regions, class_num, batch_size=5):
        roi_pool = RoIPooling(image_shape=self.__input_shape
                              , batch_size=batch_size)([fmaps, regions])

        #regs_sh = roi_pool.get_shape()
        #input_shape = (regs_sh[1], regs_sh[2], regs_sh[3], regs_sh[4])
        #flt = TimeDistributed(Flatten(), input_shape=input_shape)(roi_pool)

        flt = TimeDistributed(Flatten())(roi_pool)

        fc1 = TimeDistributed(Dense(2048))(flt)
        norm1 = TimeDistributed(BatchNormalization())(fc1)
        act1 = TimeDistributed(Activation('relu'))(norm1)
        fc2 = TimeDistributed(Dense(2048))(act1)
        norm2 = TimeDistributed(BatchNormalization())(fc2)
        act2 = TimeDistributed(Activation('relu'))(norm2)
        outputs = act2

        cls_logits = TimeDistributed(Dense(class_num))(outputs)
        clsses = Activation('softmax')(cls_logits)

        ofs_tmp = TimeDistributed(Dense(class_num * 4))(outputs)
        offsets = Reshape([-1, class_num, 4])(ofs_tmp)

        return clsses, offsets


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

    def default_compile(self):
        """
        TODO : Write description
        default_compile
        """
        self.__model.compile(optimizer=SGD(momentum=0.9, decay=0.0001)
                             , loss=[None] * len(self.__model.outputs), metrics=[])

    def get_model_with_default_compile(self):
        """
        TODO : Write description
        get_model_with_default_compile
        """
        self.default_compile()
        return self.__model

    def draw_model_summary(self, file_name='model.png'):
        plot_model(self.__model, to_file=file_name)
