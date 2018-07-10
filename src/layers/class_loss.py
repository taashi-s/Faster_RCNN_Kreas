"""
TODO : Write description
Class Loss Layer Module
"""

from keras.layers.core import Lambda

class ClassLoss():
    """
    TODO : Write description
    Class Loss Layer class
    """

    def __init__(self):
        self.layer = Lambda(self.__class_loss
                            , output_shape=self.__class_loss_output_shape)


    def __call__(self):
        return self.layer


    def __class_loss(self, inputs):
        return inputs


    def __class_loss_output_shape(self, inputs_shape):
        return inputs_shape
