from keras.layers import Dense, MaxPooling3D, Conv3D, Flatten, ConvLSTM2D
from keras.models import Sequential

def conv3D(conv_filters=(32, 64), dense_nodes=128):
    model = Sequential()

    model.add(Conv3D(conv_filters[0], (3, 3, 3), activation='relu', name="conv1", data_format='channels_last'))
    model.add(MaxPooling3D((2, 2, 2)))

    model.add(Conv3D(conv_filters[1], (3, 3, 3), activation='relu', name="conv2", data_format='channels_last'))
    model.add(MaxPooling3D((2, 2, 2)))

    model.add(ConvLSTM2D(40, (3, 3)))
    model.add(Flatten())

    model.add(Dense(dense_nodes, activation="relu"))
    model.add(Dense(5, activation='softmax'))

    model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model