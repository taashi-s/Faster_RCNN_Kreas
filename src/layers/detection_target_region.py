"""
TODO : Write description
Region proposal Layer Module
"""

import tensorflow as tf
import keras.engine.base_layer as KELayer
import keras.backend as KB

from utils.regions_utils import RegionsUtils


class DetectionTargetRegion(KELayer.Layer):
    """
    TODO : Write description
    Detection Target Region Layer class
    """

    def __init__(self, positive_threshold=0.5, positive_ratio=0.33, image_shape=None
                 , batch_size=5, exclusion_threshold=0.1, count_per_batch=64, **kwargs):
        super(DetectionTargetRegion, self).__init__(**kwargs)
        self.__th = positive_threshold
        self.__excl_th = exclusion_threshold
        self.__count_per_batch = count_per_batch
        self.__ratio = positive_ratio
        self.__image_shape = image_shape
        self.__batch_size = batch_size

    def call(self, inputs, **kwargs):
        return self.__detection_target_region(*inputs)

    def __detection_target_region(self, cls_labels, reg_labels, regions):
        input_shape_list = regions.get_shape().as_list()

        norm_reg_labels = reg_labels
        if self.__image_shape is not None:
            (img_h, img_w, _) = self.__image_shape
            norm_reg_labels = RegionsUtils(reg_labels).normalize(img_h, img_w)

        target_clss = []
        target_ofss = []
        target_regs = []

        zip_data = self.__zip_by_batch(cls_labels, norm_reg_labels, regions, self.__batch_size)
        for data in zip_data:
            data_s = self.__shaping_inputs(*data)
            target_data = self.__get_target_data(*data_s)

            target_reg, target_ofs, target_cls = target_data
            target_clss.append(tf.expand_dims(target_cls, 2))
            target_ofss.append(target_ofs)
            target_regs.append(target_reg)
        return [KB.stack(target_clss), KB.stack(target_ofss), KB.stack(target_regs)]

    def __zip_by_batch(self, cls_labels, reg_labels, regions, batch_size):
        split_cls_labels = tf.split(cls_labels, batch_size)
        split_reg_labels = tf.split(reg_labels, batch_size)
        split_regions = tf.split(regions, batch_size)
        return zip(split_cls_labels, split_reg_labels, split_regions)


    def __shaping_inputs(self, cls_label, reg_label, region):
        cls_label_2d = KB.squeeze(cls_label, 0)
        reg_label_2d = KB.squeeze(reg_label, 0)
        region_2d = KB.squeeze(region, 0)

        target_lbl_ids = KB.flatten(tf.where(KB.any(reg_label_2d, axis=1)))
        target_reg_ids = KB.flatten(tf.where(KB.any(region_2d, axis=1)))

        cls_lbl = KB.gather(cls_label_2d, target_lbl_ids)
        reg_lbl = KB.gather(reg_label_2d, target_lbl_ids)
        reg = KB.gather(region_2d, target_reg_ids)
        return cls_lbl, reg_lbl, reg


    def __get_positive(self, ious):
        max_iou = KB.max(ious, axis=1)
        ids = KB.flatten(tf.where(max_iou >= self.__th))
        count = round(self.__count_per_batch * self.__ratio)
        return self.__get_shuffle_sample(ids, count)


    def __get_negative(self, ious, positive_count):
        max_iou = KB.max(ious, axis=1)
        ids = KB.flatten(tf.where(self.__excl_th <= max_iou and max_iou < self.__th))
        count = self.__count_per_batch - positive_count
        return self.__get_shuffle_sample(ids, count)


    def __get_shuffle_sample(self, sample, count):
        sample_num = KB.shape(sample)[0]
        limit = KB.minimum(count, sample_num)
        shuffle_sample = tf.random_shuffle(sample)[:limit]
        return KB.switch(sample_num > 0, shuffle_sample, sample)


    def __get_target_data(self, cls_lbl, reg_lbl, reg):
        ious = RegionsUtils(reg).clac_iou(reg_lbl)
        positive_ids = self.__get_positive(ious)
        negative_ids = self.__get_negative(ious, KB.shape(positive_ids)[0])

        target_reg_ids = KB.concatenate((positive_ids, negative_ids))
        max_iou_ids = KB.argmax(ious, axis=1)
        target_reg_lbl_ids = KB.gather(max_iou_ids, target_reg_ids)
        target_cls_lbl_ids = KB.gather(max_iou_ids, positive_ids)

        target_reg = KB.gather(reg, target_reg_ids)
        target_ofs = self.__get_target_offset(reg_lbl, target_reg_lbl_ids, target_reg)
        target_cls = self.__get_target_class_label(cls_lbl, target_cls_lbl_ids, negative_ids)
        return self.__padding_data(target_reg, target_ofs, target_cls)


    def __get_target_offset(self, reg_lbl, target_reg_lbl_ids, target_reg):
        target_reg_lbl = KB.gather(reg_lbl, target_reg_lbl_ids)
        return RegionsUtils(target_reg_lbl).calc_offset(target_reg)


    def __get_target_class_label(self, cls_lbl, target_cls_lbl_ids, negative_ids):
        target_cls = KB.cast(KB.gather(cls_lbl, target_cls_lbl_ids), 'int32')
        padding = KB.zeros([KB.shape(negative_ids)[0]], dtype='int32')
        return KB.concatenate((target_cls, padding))


    def __padding_data(self, target_reg, target_ofs, target_cls):
        padding_count = KB.maximum(self.__count_per_batch - KB.shape(target_reg)[0], 0)
        padding_target_reg = tf.pad(target_reg, [(0, padding_count), (0, 0)])
        padding_target_ofs = tf.pad(target_ofs, [(0, padding_count), (0, 0)])
        padding_target_cls = tf.pad(target_cls, [(0, padding_count), (0, 0)])
        return padding_target_reg, padding_target_ofs, padding_target_cls


    def compute_output_shape(self, input_shape):
        return [(None, self.__count_per_batch, 1)
                , (None, self.__count_per_batch, 4)
                , (None, self.__count_per_batch, 4)
               ]
