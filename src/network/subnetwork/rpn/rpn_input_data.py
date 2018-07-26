"""
RPN Input data Utils
"""

import numpy as np
import tensorflow as tf
import keras.backend as KB

from .layers.utils import RegionsUtils

NEUTRAL_LABEL = -1
NON_OBJECT_LABEL = 0
OBJECT_LABEL = 1

def get_anchors(image_size, input_shape, scales=None, ratios=None):
    """
    TODO : Write description
    get_anchors
    """
    base_pos = __get_anchor_base_positions(scales=scales, ratios=ratios)
    shifts = __get_anchor_shifts(image_size, input_shape)
    return __create_anchors(*base_pos, *shifts)


def __get_anchor_base_positions(scales=None, ratios=None):
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


def __get_anchor_shifts(image_shape, input_shape):
    image_h, image_w, _ = image_shape
    input_h, input_w, _ = input_shape
    magni_h, magni_w = (image_h // input_h, image_w // input_w)

    shift_hs = np.arange(0, image_h, magni_h)
    shift_ws = np.arange(0, image_w, magni_w)
    return np.meshgrid(shift_ws, shift_hs)


def __create_anchors(pos_ws, pos_hs, shift_xs, shift_ys):
    widths, centers_x = np.meshgrid(pos_ws, shift_xs)
    heights, centers_y = np.meshgrid(pos_hs, shift_ys)

    centers = np.stack([centers_y, centers_x], axis=2).reshape([-1, 2])
    sizes = np.stack([heights, widths], axis=2).reshape([-1, 2])
    aaa = np.concatenate([centers - 0.5 * sizes, centers + 0.5 * sizes], axis=1).astype('float32')
    return np.concatenate([centers - 0.5 * sizes, centers + 0.5 * sizes], axis=1).astype('float32')


def make_inputs(anchors, regs, height, width
                , positive_threshold=0.5, negative_threshold=0.3
                , sample_num=60, positive_rate=0.5
                , refinement_std_dev=None):
    """
    TODO : Write description
    make_inputs
    """
    ref_sd = [0.1, 0.1, 0.2, 0.2]
    if refinement_std_dev is not None:
        ref_sd = refinement_std_dev
    anhors_num = len(anchors)

    inside_ids = np.where((anchors[:, 0] >= 0)
                          & (anchors[:, 1] >= 0)
                          & (anchors[:, 2] <= height)
                          & (anchors[:, 3] <= width)
                         )[0]
    inside_anchors = anchors[inside_ids]

    argmax_ious, cls_label = __make_cls_label(inside_anchors, regs
                                              , positive_threshold, negative_threshold
                                              , sample_num, positive_rate)
    ofs_label = RegionsUtils(inside_anchors).calc_offset_np(regs[argmax_ious])
    ofs_label /= np.array(ref_sd)
    cls_label = __unmap(cls_label, anhors_num, inside_ids, fill=-1)
    ofs_label = __unmap(ofs_label, anhors_num, inside_ids, fill=0)
    cls_label = np.expand_dims(cls_label, axis=1)
    return ofs_label, cls_label


def __make_cls_label(anchors, regs
                     , positive_threshold, negative_threshold
                     , sample_num, positive_rate):

    cls_lbl = np.full((len(anchors)), NEUTRAL_LABEL)
    argmax_ious_per_anchor, max_ious, argmax_ious = __calc_ious(anchors, regs)

    cls_lbl[argmax_ious] = OBJECT_LABEL
    cls_lbl[max_ious >= positive_threshold] = OBJECT_LABEL
    cls_lbl[max_ious < negative_threshold] = NON_OBJECT_LABEL

    positive_max = int(positive_rate * sample_num)
    cls_lbl = __restrict_cls_label(cls_lbl, positive_max, OBJECT_LABEL)

    negative_max = sample_num - np.sum(cls_lbl == OBJECT_LABEL)
    cls_lbl = __restrict_cls_label(cls_lbl, negative_max, NON_OBJECT_LABEL)

    return argmax_ious_per_anchor, cls_lbl


def __restrict_cls_label(cls_lbl, limit, target_label_val):
    indexes = np.where(cls_lbl == target_label_val)[0]
    over_count = len(indexes) - limit
    if over_count > 0:
        disable_index = np.random.choice(indexes, size=(over_count), replace=False)
        cls_lbl[disable_index] = NEUTRAL_LABEL
    return cls_lbl


def __calc_ious(anchors, regs):
    ious = RegionsUtils(anchors).calc_iou_np(regs)
    argmax_ious_per_anchor = ious.argmax(axis=1)
    max_ious = ious[np.arange(len(ious)), argmax_ious_per_anchor]
    argmax_ious = np.where(ious == ious.max())[0]
    return argmax_ious_per_anchor, max_ious, argmax_ious


def __unmap(data, count, index, fill=0):
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=data.dtype)
        ret.fill(fill)
        ret[index] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[index, :] = data
    return ret
