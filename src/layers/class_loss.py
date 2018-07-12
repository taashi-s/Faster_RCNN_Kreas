"""
TODO : Write description
Class Loss Layer Module
"""

from keras.layers.core import Lambda
import keras.backend as KB
import utils.loss_utils as lu

class ClassLoss():
    """
    TODO : Write description
    Class Loss Layer class
    """

    def __init__(self):
        self.layer = Lambda(lambda inputs: self.__class_loss(*inputs)
                            , output_shape=self.__class_loss_output_shape)


    def __call__(self):
        return self.layer


    def __class_loss(self, label, pred):
        return lu.class_labels_mean_loss(KB.squeeze(label, 2), pred)


    def __class_loss_output_shape(self, _):
        return [1]
