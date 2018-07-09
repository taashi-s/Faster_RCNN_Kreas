"""
Loss Utils modeule
"""

import keras.backend as KB
import tensorflow as tf

def class_labels_mean_loss(labels, preds):
    """
    Class Labels mean loss
    """
    loss = KB.sparse_categorical_crossentropy(labels, preds)
    return KB.switch(tf.size(loss) > 0, KB.mean(loss), KB.constant(0.0))


def offset_labels_mean_loss(labels, preds):
    """
    Offset Labels mean loss
    """
    loss = smooth(labels, preds)
    return KB.switch(tf.size(loss) > 0, KB.mean(loss), KB.constant(0.0))


def smooth(labels, preds):
    """
    smooth
    """
    diff = KB.abs(labels - preds)
    less = KB.cast(KB.less(diff, 1.0), "float32")
    return (less * 0.5 * diff**2) + (1 - less) * (diff - 0.5)
