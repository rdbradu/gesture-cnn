import tensorflow as tf

from tensorflow import keras

from keras import layers
from keras.layers import Dense, MaxPooling3D, Conv3D, Flatten, ConvLSTM2D, BatchNormalization, Dropout, RandomFlip, RandomRotation, LSTM, Reshape
from keras.models import Sequential
from keras.optimizers import SGD, Adam

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

from process import extract_frames
from augment import VideoAugmentation
from dataset import extract_dataset

def conv3DLSTM(conv_filters=(32, 64, 128), dense_nodes=(256, 128)):
    model = Sequential()

    model.add(VideoAugmentation(input_shape=(30, 112, 112, 3)))

    model.add(Conv3D(conv_filters[0], (3, 3, 3), activation='relu', name="conv3d1", strides=(1, 2, 2), padding="same"))
    model.add(MaxPooling3D((1, 2, 2), strides=(1, 2, 2)))

    model.add(Conv3D(conv_filters[1], (3, 3, 3), activation='relu', name="conv3d2", padding="same"))
    model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2)))

    model.add(Conv3D(conv_filters[2], (3, 3, 3), activation='relu', name="conv3d3a", padding="same"))
    model.add(Conv3D(conv_filters[2], (3, 3, 3), activation='relu', name="conv3d3b", padding="same"))

    model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2)))

    model.add(Conv3D(conv_filters[2], (3, 3, 3), activation='relu', name="conv3d4a", padding="same"))
    model.add(Conv3D(conv_filters[2], (3, 3, 3), activation='relu', name="conv3d4b", padding="same"))

    model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2)))

    model.add(BatchNormalization())
    model.add(Flatten())

    model.add(Reshape((9, 384))) 

    model.add(LSTM(50))
    model.add(Dropout(0.2))

    model.add(Dense(dense_nodes[0], activation='relu'))
    model.add(Dense(dense_nodes[1], activation='relu'))

    model.add(Dense(5, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.00001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    # extract_dataset("./20bnjester/Train/", "./trainset/", "./20bnjester/Train.csv", "train")
    # extract_dataset("./20bnjester/Validation/", "./validationset/", "./20bnjester/Validation.csv", "validation")

    train_x, train_y = extract_frames("train.csv", "./trainset/")
    val_x, val_y = extract_frames("validation.csv", "./validationset/")

    train_x = np.array(train_x)
    val_x = np.array(val_x)

    train_y = np.array(train_y)
    val_y = np.array(val_y)

    # scaler = StandardScaler()
    # scaled_train_x  = scaler.fit_transform(train_x.reshape(-1, 30*112*112))
    # scaled_train_x  = scaled_train_x.reshape(-1, 30, 112, 112, 3)

    # scaled_val_x  = scaler.fit_transform(val_x.reshape(-1, 30*112*112))
    # scaled_val_x  = scaled_val_x.reshape(-1, 30, 112, 112, 3)

    # train_x = np.array(scaled_train_x)
    # train_y = np.array(train_y)

    # val_x = np.array(scaled_val_x)
    # val_y = np.array(val_y)

    nn = conv3DLSTM()

    history = nn.fit(train_x, train_y, validation_data=(val_x, val_y), batch_size=32, epochs=15)
    nn.save("3dcnn_lstm.h5")
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