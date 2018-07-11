"""
TODO : Write description
Region proposal Region Loss Layer Module
"""

from keras.layers.core import Lambda

class RPRegionLoss():
    """
    TODO : Write description
    Region proposal Region Loss Layer class
    """

    def __init__(self):
        self.layer = Lambda(self.__region_loss
                            , output_shape=self.__region_loss_output_shape)


    def __call__(self):
        return self.layer


    def __region_loss(self, inputs):
        return inputs


    def __region_loss_output_shape(self, inputs_shape):
        return inputs_shape
