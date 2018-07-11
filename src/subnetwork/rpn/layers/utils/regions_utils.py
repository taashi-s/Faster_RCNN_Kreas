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
        regions = self.__regions
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
