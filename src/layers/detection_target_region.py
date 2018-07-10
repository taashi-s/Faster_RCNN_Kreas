"""
TODO : Write description
Region Proporsal Layer Module
"""

import keras.engine.base_layer as KELayer


class DetectionTargetRegion(KELayer.Layer):
    """
    TODO : Write description
    Detection Target Region Layer class
    """

    def __init__(self):
        super().__init__()


    def build(self, input_shape):
        super().build(input_shape)


    def call(self, inputs, **kwargs):
        return inputs


    def compute_output_shape(self, input_shape):
        return input_shape
