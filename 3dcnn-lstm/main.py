import tensorflow as tf

from tensorflow import keras

from keras import layers
from keras.layers import Dense, MaxPooling3D, Conv3D, Flatten, ConvLSTM2D, BatchNormalization, Dropout, RandomFlip, RandomRotation
from keras.models import Sequential
from keras.optimizers import SGD

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

from process import extract_frames
from dataset import extract_dataset

def conv3D(conv_filters=(32, 64), dense_nodes=(128, )):
    model = Sequential()

    model.add(Conv3D(conv_filters[0], (3, 3, 3), activation='relu', name="conv1", data_format='channels_last'))
    model.add(MaxPooling3D((2, 2, 2), data_format='channels_last'))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())

    model.add(Conv3D(conv_filters[1], (3, 3, 3), activation='relu', name="conv2", data_format='channels_last'))
    model.add(MaxPooling3D((2, 2, 2), data_format='channels_last'))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_sequences=True, dropout=0.1))
    model.add(ConvLSTM2D(filters=20, kernel_size=(3, 3), padding='same', dropout=0.1))

    model.add(Flatten())

    model.add(Dense(dense_nodes[0], activation="relu"))
    model.add(Dense(4, activation='softmax'))

    model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    extract_dataset("./20bnjester/Train/", "./trainset/", "./20bnjester/Train.csv", "train")
    extract_dataset("./20bnjester/Validation/", "./validationset/", "./20bnjester/Validation.csv", "validation")
    train_x, train_y = extract_frames("train.csv", "./trainset/")
    val_x, val_y = extract_frames("validation.csv", "./validationset/")

    #! IMAGE VISUAL

    # fig = plt.figure()
    # for i in range(2,4):
    #     for num,frame in enumerate(train_x[i][0:18]):
    #         y = fig.add_subplot(4,5,num+1)
    #         y.imshow(frame, cmap='gray')
    #     fig = plt.figure()
    # plt.show()

    train_x = np.array(train_x, dtype=np.float32)
    val_x = np.array(val_x, dtype=np.float32)

    #! RESHAPE?

    scaler = StandardScaler()
    scaled_train_x  = scaler.fit_transform(train_x.reshape(-1, 15*64*64))
    scaled_train_x  = scaled_train_x.reshape(-1, 15, 64, 64, 1)

    scaled_val_x  = scaler.fit_transform(val_x.reshape(-1, 15*64*64))
    scaled_val_x  = scaled_val_x.reshape(-1, 15, 64, 64, 1)

    train_x = np.array(scaled_train_x)
    train_y = np.array(train_y)

    val_x = np.array(scaled_val_x)
    val_y = np.array(val_y)

    train_dataset = tf.data.Dataset.from_tensor_slices((scaled_train_x, train_y))
    val_dataset = tf.data.Dataset.from_tensor_slices((scaled_val_x, val_y))

    cnn = conv3D()
    cnn.summary()

    history = cnn.fit(train_x, train_y, validation_data=(val_x, val_y), batch_size=16, epochs=10)
    cnn.save("3dcnn_lstm.h5")
    plt.plot(history.history["val_accuracy"])
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_loss"])
    plt.plot(history.history["loss"])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['val_accuracy', 'accuracy', 'val_loss', 'loss'], loc='upper left')
    plt.show()

main()