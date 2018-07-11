"""
TODO : Write description
Non Maximal Suppression Module
"""

import tensorflow as tf

class NMS():
    """
    TODO : Write description
    Non Maximal Suppression Class
    """

    def __init__(self, count_limit, iou_threshold):
        self.__limit = count_limit
        self.__th = iou_threshold


    def __call__(self, regions, scores):
        return self.__non_maximal_supperssion(regions, scores)


    def __non_maximal_supperssion(self, regions, scores):
        ids = tf.image.non_max_suppression(regions, scores, self.__limit
                                           , iou_threshold=self.__th)
        regs = tf.gather(regions, ids)
        padding = tf.maximum(self.__limit - tf.shape(regs)[0], 0)
        padding_regs = tf.pad(regs, [(0, padding), (0, 0)])
        return padding_regs


    def nms(self, regions, scores):
        """
        TODO : Write description
        nms
        """
        return self.__non_maximal_supperssion(regions, scores)
