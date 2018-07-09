"""
TODO : Write description
Region Proporsal Network Module
"""

from keras.models import Model
from keras.engine.topology import Input
from keras.layers.convolutional import Conv2D
# from keras.layers.normalization import BatchNormalization

from layers.region_proporsal import RegionProporsal

class RegionProporsalNet():
    """
    TODO : Write description
    RegionProporsalNet class
    """

    def __init__(self, input_shape, input_layers=None, box_by_anchor=9, trainable=True):
        self.__trainable = trainable
        self.__input_shape = input_shape

        inputs = Input(self.__input_shape)
        if input_layers is not None:
            inputs = input_layers

        intermediate = Conv2D(256, 3, strides=3, activation="relu"
                              , trainable=self.__trainable)(inputs)

        cls_layer = Conv2D(2 * box_by_anchor, 1, activation="relu"
                           , trainable=self.__trainable)(intermediate)
        reg_layer = Conv2D(4 * box_by_anchor, 1, activation="relu"
                           , trainable=self.__trainable)(intermediate)

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
