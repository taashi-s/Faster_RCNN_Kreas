"""
TODO : Write description
Region Proporsal Network Module
"""

from keras.models import Model
from keras.engine.topology import Input
from keras.layers.core import Flatten, Dense, Reshape, Dropout, Activation, Lambda
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization

from layers.region_proporsal import RegionProporsal

class RPN():
    """
    TODO : Write description
    RPN class
    """

    def __init__(self, input_shape, box_by_anchor=9):
        self.__input_shape = input_shape

        inputs = Input(self.__input_shape)

        intermediate = Conv2D(256, 3, strides=3, activation="relu")(inputs)

        cls_layer = Conv2D(2 * box_by_anchor, 1, activation="relu")(intermediate)
        reg_layer = Conv2D(4 * box_by_anchor, 1, )(intermediate)

        outputs = RegionProporsal()([reg_layer, cls_layer])

        self.__network = outputs

        self.__model = Model(inputs=[inputs], outputs=[outputs])

    def get_input_size(self):
        """
        TODO : Write description
        get_model
        """
        return self.__input_shape

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
