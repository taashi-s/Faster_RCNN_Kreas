"""
TODO : Write description
Faster R-CNN Module
"""

from keras.models import Model
from keras.engine.topology import Input
from keras.layers.core import Flatten, Dense, Reshape, Dropout, Activation
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from ResNet.resnet import ResNet

class FasterRCNN():
    """
    TODO : Write description
    ResNet class
    """

    def __init__(self, input_shape, channel_width=10):
        self.__input_shape = input_shape

        inputs = Input(self.__input_shape)


        outputs = Dense(1000, activation="softmax")(inputs)
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
