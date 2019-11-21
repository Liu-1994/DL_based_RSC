from keras import models
from keras.layers import *


def cnn_model(num_classes):
    dr = 0.5
    model = models.Sequential()
    model.add(Conv2D(32, (2, 2), padding='valid', activation="relu", input_shape=[1024, 2, 1]))
    # model.add(Dropout(dr))
    model.add(Reshape([1023, 32]))
    model.add(Conv1D(64, 3, strides=2, padding="valid", activation="relu"))
    # model.add(Dropout(dr))
    model.add(Conv1D(128, 3, strides=2, padding="valid", activation="relu"))
    # model.add(Dropout(dr))
    model.add(Conv1D(256, 3, strides=2, padding="valid", activation="relu"))
    # model.add(Dropout(dr))
    model.add(Conv1D(128, 3, strides=2, padding="valid", activation="relu"))
    # model.add(Dropout(dr))
    model.add(Conv1D(64, 3, strides=2, padding="valid", activation="relu"))
    # model.add(Dropout(dr))
    model.add(Conv1D(32, 3, strides=2, padding="valid", activation="relu"))
    # model.add(Dropout(dr))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(dr))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    model.summary()
    return model
