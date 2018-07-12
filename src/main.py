""" main """

from keras.optimizers import SGD
import numpy as np
from faster_rcnn import FasterRCNN, TrainTarget

INPUT_SHAPE = (2048, 2048, 3)
BATCH_SIZE = 256
EPOCHS = 100

DIR_MODEL = '.'
FILE_MODEL = 'ResNetModel.hdf5'

def train():
    """ train """
    print('execute train')

    # TODO
    train_inputs = None
    train_teachers = None
    test_inputs = None
    test_teachers = None

    anchors = get_default_anchors()
    train_taegets=[TrainTarget.BACKBONE, TrainTarget.RPN, TrainTarget.HEAD]

    network = FasterRCNN(INPUT_SHAPE, 2, anchors, train_taegets=train_taegets)
    model = network.get_model()
    model.compile(optimizer=SGD(momentum=0.9, decay=0.0001)
                  , loss=[], metrics=[])
#    his = model.fit(train_inputs, train_teachers
#                    , batch_size=BATCH_SIZE
#                    , epochs=EPOCHS
#                    , validation_data=(test_inputs, test_teachers)
#                    , verbose=1)
#    model.save_weights(os.path.join(DIR_MODEL, FILE_MODEL))

def predict():
    """ predict """
    print('execute predict')

def get_default_anchors():
    return np.zeros([60000*9, 4], dtype='int32')

if __name__ == '__main__':
    train()
    predict()
