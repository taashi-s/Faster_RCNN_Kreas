"""
TODO : Write description
Region Proporsal Layer Module
"""

from keras.models import Model
from keras.layers.core import Reshape, Lambda


class RegionProporsal():
    """
    TODO : Write description
    Region Proporsal Layer class
    """

    def __init__(self):
        self.layer = Lambda(self.__region_proporsal
                            , output_shape=self.__region_proporsal_output_shape)


    def __call__(self):
        return self.layer


    def __region_proporsal(self, inputs):
        return inputs


    def __region_proporsal_output_shape(self, inputs_shape):
        return inputs_shape
