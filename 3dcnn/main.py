import os
import csv
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2

from tensorflow import keras
from keras import layers
from keras.layers import Dense, MaxPooling3D, Conv3D, Flatten, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical

import matplotlib.pyplot as plt

#! BASIC 3DCNN IMPLEMENTATION

#! IMAGES ARE BEING READ IN 3 CHANNELS (RGB)

#! Loads train_set and test_set representation
#! as pandas dataframe

def get_set_frames():
    train_df = pd.read_csv("./dataset/train.csv", sep=";", header=None)
    train_df.rename(columns={0: "Image", 1: "Label", 2: "ImageNumber"}, inplace=True)
    train_df.info(memory_usage="deep")
    test_df = pd.read_csv("./dataset/val.csv", sep=";", header=None)
    test_df.rename(columns={0: "Image", 1: "Label", 2: "ImageNumber"}, inplace=True)
    test_df.info(memory_usage="deep")

    print(f"Total videos for testing: {len(test_df)}")
    print(f"Total videos for training: {len(train_df)}")

    return train_df, test_df

#! Utilities to open video files using CV2

#* Gets number of height and width pixels
#* factors a new starting point for a cropped image 

def crop_center_square(frame):
    y, x = frame.shape[0: 2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)

    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

#! Resize image (crop center of image) 
#! from initial resolution to 224 x 224 
#! because initial image resolution is pretty small

#* increase resolution

def load_image(df, path):
    labels = df["Label"].values.tolist()
    img = cv2.imread(path)

    frames = crop_center_square(img)

    frames = cv2.resize(frames, (224, 224))
    return frames, labels

def load_gestures(df):
    gesture_number = to_categorical(df["ImageNumber"])
    video_paths = df["Image"].values.tolist()

    return gesture_number, video_paths

#! normalize results from all images
#! by dividing to /255.0 

#? pixels are represented as 0 to 255 integers
#? that means you would send pixel values
#? as values of [0, 1] 

def get_video_set_data(videos, df, directory, size):
    video_data_train = []
    for vid in videos[:size]:
        current_directory = directory + f'/{vid}'
        frame = []
        for images in os.listdir(current_directory):
            if images.endswith(".png"):
                frames, labels = load_image(df, current_directory + f'/{images}')
                frames = frames / 255.0
                frame.append(frames)
        video_data_train.append(frame)

    return video_data_train, labels

def get_basic_3dcnn(train_data_tensor):
    input_shape = (train_data_tensor.shape[1], train_data_tensor.shape[2], train_data_tensor.shape[3], train_data_tensor.shape[4])

    model = Sequential([
        #! 1ST LAYER (8 convolution filters with a kernel-size of 3x3x3)
        Conv3D(8,(3,3,3), activation='relu', input_shape=input_shape),
        MaxPooling3D((2,2,2), padding='same'),

        #! 2ND LAYER (12 convolution filters of size (3x3x3))
        Conv3D(12, (3,3,3), activation='relu'),
        MaxPooling3D(pool_size=(1,2,2), padding='same'),

        #! 3RD LAYER (32 convolution filters of size (3x3x3))
        Conv3D(32, (3,3,3), activation='relu'),
        MaxPooling3D(pool_size=(1,2,2), padding='same'),

        #! CLASSIFICATION PREPARATION
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),

        Dense(64, activation='relu'),
        Dropout(0.3),

        Dense(5, activation='softmax')
    ])
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
    return model    

def get_final_model_trainset():
    train_frame, test_frame = get_set_frames()
    train_gestures, train_videos = load_gestures(train_frame)
    test_gestures, test_videos = load_gestures(test_frame)

    train_set_data, labels = get_video_set_data(train_videos, train_frame, "./dataset/train", 150)
    
    vid_train_formatted = np.array(train_set_data).reshape(150, 30, 224, 224, 3)
    train_data, train_labels = vid_train_formatted, labels

    test_set_data, labels = get_video_set_data(test_videos, test_frame, "./dataset/val", 30)
    
    vid_test_formatted = np.array(test_set_data).reshape(30, 30, 224, 224, 3)
    test_data, test_labels = vid_test_formatted, labels

    return (test_data, test_gestures, train_data, train_gestures)

#! ACTUAL UTIL FOR CREATING A 3DCNN
#! DEFAULT CONFIGURED FOR HRN
def conv3D(conv_filters=(16, 32, 64, 128), dense_nodes=(256, 128), dropout=0.25, height=224, width=224, num_images=30):
    model = Sequential()

    model.add(Conv3D(conv_filters[0], (3, 3, 3), activation='relu', padding='same', input_shape=(num_images, height, width, 3)))
    model.add(MaxPooling3D((2, 2, 2)))

    model.add(Conv3D(conv_filters[1], (3, 3, 3), activation='relu', padding='same', input_shape=(num_images, height, width, 3)))
    model.add(MaxPooling3D((2, 2, 2)))

    model.add(Conv3D(conv_filters[2], (3, 3, 3), activation='relu', padding='same', input_shape=(num_images, height, width, 3)))
    model.add(MaxPooling3D((2, 2, 2)))

    model.add(Conv3D(conv_filters[3], (3, 3, 3), activation='relu', padding='same', input_shape=(num_images, height, width, 3)))
    model.add(MaxPooling3D((2, 2, 2)))

    model.add(Flatten())
    model.add(Dense(dense_nodes[0], activation="relu"))

    model.add(Dense(dense_nodes[1], activation="relu"))

    model.add(Dense(5, activation='softmax'))
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_basic_model(train_data, train_gestures):
    train_data = tf.convert_to_tensor(train_data)

    #! TODO - PLOT LOSS AND ACCURACY 

    model = get_basic_3dcnn(train_data)
    model.fit(train_data, train_gestures[:300], epochs=5)
    model.save("basic_3dcnn.h5")

def train_3dcnn(train_data, train_gestures):

    train_data = tf.convert_to_tensor(train_data)
    model = conv3D()
    print(len(train_data))
    print(len(train_gestures))
    history = model.fit(train_data, train_gestures[:150], epochs=5)
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["loss"])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['accuracy', 'loss'], loc='upper left')
    plt.show()
    model.save("3dcnn.h5")

def main():
    trainset = get_final_model_trainset()
    train_3dcnn(trainset[2][:150], trainset[3][:150])

main()