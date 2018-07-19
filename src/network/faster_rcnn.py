"""
TODO : Write description
Faster R-CNN Module
"""

from enum import Enum
import numpy as np
import tensorflow as tf
import keras.backend as KB
from keras.models import Model
from keras.engine.topology import Input
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, Flatten, Activation, Reshape, Lambda
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.utils import plot_model

from .subnetwork import ResNet, RegionProposalNet
from .layers import DetectionTargetRegion, RoIPooling
from .layers import RPClassLoss, RPRegionLoss, ClassLoss, RegionLoss

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

    def __init__(self, input_shape, class_num, anchors=None
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

        backbone = ResNet(inputs_images.get_shape(), input_layers=inputs_images
                          , trainable=train_backbone
                         ).get_residual_network()
        self.__backbone_network = (inputs, backbone)

        self.__anchors = anchors
        if self.__anchors is None:
            self.__anchors = self.get_anchors()
        rpn = RegionProposalNet(backbone.get_shape(), self.__anchors, input_layers=inputs_images
                                , image_shape=self.__input_shape, prev_layers=backbone
                                , batch_size=batch_size, is_predict=is_predict
                                , trainable=train_rpn
                               ).get_network()
        rpn_cls_probs, rpn_regions, rpn_prop_regs = rpn
        self.__rpn_network = (inputs, rpn)

        self.__rpn_loss_network = None
        if train_rpn and not is_predict:
            inputs_rp_cls = Input(shape=[None, 1], dtype='int32')
            inputs_rp_reg = Input(shape=[None, 4], dtype='float32')
            inputs += [inputs_rp_cls, inputs_rp_reg]

            rp_cls_losses = RPClassLoss()([inputs_rp_cls, rpn_cls_probs])
            rp_reg_losses = RPRegionLoss()([inputs_rp_cls, inputs_rp_reg, rpn_regions])
            outputs += [rp_cls_losses, rp_reg_losses]
            self.__rpn_loss_network = (inputs, outputs)

        self.__head_network = None
        self.__head_loss_network = None
        if train_head and not is_predict:
            inputs_cls = Input(shape=[None, 1], dtype='int32')
            inputs_reg = Input(shape=[None, 4], dtype='float32')
            inputs += [inputs_cls, inputs_reg]

            dtr = DetectionTargetRegion(positive_threshold=0.5, positive_ratio=0.33
                                        , image_shape=self.__input_shape, batch_size=batch_size
                                        , exclusion_threshold=0.1, count_per_batch=64
                                       )([inputs_cls, inputs_reg, rpn_prop_regs])
            dtr_cls_labels, dtr_offsets_labels, dtr_regions = dtr
            clsses, offsets = self.head_net(backbone, dtr_regions, class_num
                                            , batch_size=batch_size)
            self.__head_network = (inputs, [clsses, offsets])

            cls_losses = ClassLoss()([dtr_cls_labels, clsses])
            reg_losses = RegionLoss()([dtr_cls_labels, dtr_offsets_labels, offsets])
            outputs += [cls_losses, reg_losses]
            self.__head_loss_network = (inputs, outputs)

        if is_predict:
            clsses, offsets = self.head_net(backbone, rpn_prop_regs, class_num
                                            , batch_size=batch_size)
            self.__head_network = (inputs, [clsses, offsets])
            outputs = [rpn_prop_regs, clsses, offsets]

        self.__network = (inputs, outputs)
        self.__model = Model(inputs=inputs, outputs=outputs)

        if not is_predict:
            for output in outputs:
                self.__model.add_loss(tf.reduce_mean(output))
        else:
            dummy_loss = Lambda(lambda x: KB.constant(0.0))([inputs_images])
            self.__model.add_loss(tf.reduce_mean(dummy_loss))


    def head_net(self, fmaps, regions, class_num, batch_size=5):
        """
        TODO : Write description
        head_net
        """

        roi_pool = RoIPooling(image_shape=self.__input_shape
                              , batch_size=batch_size)([fmaps, regions])
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


    def get_backbone_network(self):
        """
        TODO : Write description
        get_backbone_network
        """
        return self.__backbone_network


    def get_rpn_network(self):
        """
        TODO : Write description
        get_rpn_network
        """
        return self.__rpn_network


    def get_rpn_loss_network(self):
        """
        TODO : Write description
        get_rpn_loss_network
        """
        return self.__rpn_loss_network


    def get_head_network(self):
        """
        TODO : Write description
        get_head_network
        """
        return self.__head_network


    def get_head_loss_network(self):
        """
        TODO : Write description
        get_head_loss_network
        """
        return self.__head_loss_network


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


    def get_anchors(self, scales=None, ratios=None):
        """
        TODO : Write description
        get_anchors
        """
        base_pos = self.__get_anchor_base_positions(scales=scales, ratios=ratios)
        shifts = self.__get_anchor_shifts()
        return self.__create_anchors(*base_pos, *shifts)


    def __calculate_backbone_reduction_ratio(self):
        image_h, image_w, _ = self.__input_shape
        _, backbone_h, backbone_w, _ = (self.__backbone_network[1]).get_shape().as_list()
        return (image_h // backbone_h, image_w // backbone_w)


    def __get_anchor_base_positions(self, scales=None, ratios=None):
        anchor_scales = (32, 64, 128)
        if scales is not None:
            anchor_scales = scales
        anchor_ratios = [0.5, 1, 2]
        if ratios is not None:
            anchor_ratios = ratios

        scales_m, ratios_m = np.meshgrid(np.array(anchor_scales), np.array(anchor_ratios))

        scales_f = scales_m.flatten()
        ratios_f = ratios_m.flatten()

        pos_hs = scales_f / np.sqrt(ratios_f)
        pos_ws = scales_f * np.sqrt(ratios_f)

        return pos_hs, pos_ws


    def __get_anchor_shifts(self):
        image_h, image_w, _ = self.__input_shape
        magni_h, magni_w = 1, 1 #self.__calculate_backbone_reduction_ratio()

        shift_hs = np.arange(0, image_h, magni_h)
        shift_ws = np.arange(0, image_w, magni_w)
        return np.meshgrid(shift_ws, shift_hs)


    def __create_anchors(self, pos_ws, pos_hs, shift_xs, shift_ys):
        widths, centers_x = np.meshgrid(pos_ws, shift_xs)
        heights, centers_y = np.meshgrid(pos_hs, shift_ys)

        centers = np.stack([centers_y, centers_x], axis=2).reshape([-1, 2])
        sizes = np.stack([heights, widths], axis=2).reshape([-1, 2])
        return np.concatenate([centers - 0.5 * sizes, centers + 0.5 * sizes], axis=1)


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
        """
        TODO : Write description
        draw_model_summary
        """
        plot_model(self.__model, to_file=file_name)
