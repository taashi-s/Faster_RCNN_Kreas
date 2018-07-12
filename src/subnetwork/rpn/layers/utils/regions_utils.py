"""
TODO : Write description
Regions Utils Module
"""

import keras.backend as KB

class RegionsUtils():
    """
    TODO : Write description
    Regions Utils Class
    """

    def __init__(self, regions):
        self.__regions = regions


    def normalize(self, max_h, max_w):
        """
        TODO : Write description
        normalize
        """
        return self.__regions / KB.variable([max_h, max_w, max_h, max_w])


    def offset(self, offsets):
        """
        TODO : Write description
        offset
        """
        regions = KB.cast(self.__regions, 'float32')
        reg_h = regions[:, 2] - regions[:, 0]
        reg_w = regions[:, 3] - regions[:, 1]

        pos_h = KB.exp(offsets[:, 2]) * reg_h
        pos_w = KB.exp(offsets[:, 3]) * reg_w
        center_y = offsets[:, 0] * reg_h + regions[:, 0] + 0.5 * reg_h
        center_x = offsets[:, 1] * reg_w + regions[:, 1] + 0.5 * reg_w

        min_x = center_x - 0.5 * pos_w
        min_y = center_y - 0.5 * pos_h
        max_x = center_x + 0.5 * pos_w
        max_y = center_y + 0.5 * pos_h
        return KB.transpose(KB.stack((min_y, min_x, max_y, max_x), axis=0))


    def calc_iou(self, target_regions):
        """
        TODO : Write description
        calc_iou
        """
        pos_tl = KB.maximum(self.__regions[:, None, :2], target_regions[:, :2])
        pos_br = KB.maximum(self.__regions[:, None, 2:], target_regions[:, 2:])
        t_p = KB.prod(pos_br - pos_tl, axis=2) * KB.cast(KB.all(pos_br > pos_tl, axis=2), 'float32')
        g_t = KB.prod(self.__regions[:, 2:] - self.__regions[:, :2], axis=1)
        p_r = KB.prod(target_regions[:, 2:] - target_regions[:, :2], axis=1)
        return t_p / (g_t + p_r - t_p)


    def calc_offset(self, target_region):
        """
        TODO : Write description
        calc_offset
        """
        base_region = self.__regions
        height = base_region[:, 2] - base_region[:, 0]
        width = base_region[:, 3] - base_region[:, 1]
        ctr_y = base_region[:, 0] + 0.5 * height
        ctr_x = base_region[:, 1] + 0.5 * width

        target_height = target_region[:, 2] - target_region[:, 0]
        target_width = target_region[:, 3] - target_region[:, 1]
        target_ctr_y = target_region[:, 0] + 0.5 * target_height
        target_ctr_x = target_region[:, 1] + 0.5 * target_width

        height = KB.maximum(height, KB.epsilon())
        width = KB.maximum(width, KB.epsilon())

        pos_y = (target_ctr_y - ctr_y) / height
        pos_x = (target_ctr_x - ctr_x) / width
        len_h = KB.log(target_height / height)
        len_w = KB.log(target_width / width)

        return KB.transpose(KB.stack((pos_y, pos_x, len_h, len_w), axis=0))
