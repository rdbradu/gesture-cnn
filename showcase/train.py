import numpy as np
import os
import datetime

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Flatten, TimeDistributed, Flatten, BatchNormalization, MaxPooling2D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from images import generate
from plot import plot_loss_accuracy

import tensorflow as tf
print(tf.version)
np.random.seed(30)
import random as rn
rn.seed(30)
from keras import backend as K
tf.random.set_seed(30)

def calculate_steps(num_train_sequences, num_val_sequences, batch_size):
    if (num_train_sequences%batch_size) == 0:
        steps_per_epoch = int(num_train_sequences/batch_size)
    else:
        steps_per_epoch = (num_train_sequences//batch_size) + 1

    if (num_val_sequences%batch_size) == 0:
        validation_steps = int(num_val_sequences/batch_size)
    else:
        validation_steps = (num_val_sequences//batch_size) + 1

    return steps_per_epoch, validation_steps


def model_callbacks(folder_name):
    curr_dt_time = datetime.datetime.now()
    model_name = str(folder_name) + '_' + str(curr_dt_time).replace(' ','').replace(':','_') + '/'

    if not os.path.exists(model_name):
        os.mkdir(model_name)

    filepath = model_name + 'model-{epoch:05d}.h5'

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

    LR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=0.0001, cooldown=1, verbose=1)

    return [checkpoint, LR]

from keras.applications import mobilenet

def mobilenet_RNN(cells=128, dense_nodes=128, dropout=0.25, num_images=20, height=120, width=120, num_classes=5):

    mobilenet_transfer = mobilenet.MobileNet(weights='imagenet', include_top=False)

    model = Sequential()
    model.add(TimeDistributed(mobilenet_transfer,input_shape=(num_images, height, width, 3)))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Flatten()))

    model.add(LSTM(cells))
    model.add(Dropout(dropout))
    model.add(Dense(dense_nodes, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation='softmax'))

    opt = tf.keras.optimizers.Adam()
    model.compile(opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model

def main():
    num_epochs = 15
    batch_size = 32
    num_frames = 20

    model = mobilenet_RNN(num_images=num_frames)

    callbacks_list = model_callbacks("model_retrain_mobilenet_gru")

    train_path = 'dataset/train'
    val_path = 'dataset/val'

    train_doc = np.random.permutation(open('dataset/train.csv').readlines())
    val_doc = np.random.permutation(open('dataset/val.csv').readlines())

    num_train_sequences = len(train_doc)
    num_val_sequences = len(val_doc)

    steps_per_epoch, validation_steps = calculate_steps(num_train_sequences, num_val_sequences, batch_size)

    train_generator = generate(train_path, train_doc, batch_size, num_images=num_frames, augment=True)

    val_generator   = generate(val_path, val_doc, batch_size, num_images=num_frames)

    history = model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=num_epochs, verbose=1,
                              callbacks=callbacks_list, validation_data=val_generator,
                              validation_steps=validation_steps, class_weight=None, 
                              workers=1, initial_epoch=0)
    
    plot_loss_accuracy(history)