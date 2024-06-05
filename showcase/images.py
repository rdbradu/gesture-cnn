#! PREPROCESS + AUGMENT

import os
import numpy as np
import imageio
from skimage.transform import resize
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    zoom_range=0.1,
    zca_whitening=True,
    width_shift_range=0.1,
    height_shift_range=0.1)

def generate(source_path, folder_list, batch_size, num_images=20, height=120, width=120, augment=False):
    total_frames = 30

    while True:
        t = np.random.permutation(folder_list)
        num_batches = len(t) // batch_size
        img_idx = np.round(np.linspace(0,total_frames-1,num_images)).astype(int)

        for batch in range(num_batches):
            batch_data   = np.zeros((batch_size, num_images, height, width, 3))
            batch_labels = np.zeros((batch_size, 5))

            for folder in range(batch_size):
                imgs = os.listdir(source_path+'/'+ t[folder + (batch*batch_size)].split(';')[0])
                for idx,item in enumerate(img_idx):
                    image = imageio.imread(source_path+'/'+ t[folder + (batch*batch_size)].strip().split(';')[0]+'/'+imgs[item]).astype(np.float32)

                    h, w, c = image.shape
                    image = resize(image, (height, width), anti_aliasing=True)

                    if augment:
                        if np.random.randn() > 0:
                            image = datagen.random_transform(image)

                    batch_data[folder,idx,:,:,0] = (image[...,0])/255 # NORMALIZE ON EACH CHANNEL
                    batch_data[folder,idx,:,:,1] = (image[...,1])/255 # NORMALIZE ON EACH CHANNEL
                    batch_data[folder,idx,:,:,2] = (image[...,2])/255 # NORMALIZE ON EACH CHANNEL

                batch_labels[folder, int(t[folder + (batch*batch_size)].strip().split(';')[2])] = 1

            yield batch_data, batch_labels

        if (len(t)%batch_size):
            remaining_batch_size = len(t)%batch_size
            batch_data   = np.zeros((remaining_batch_size, num_images, height, width, 3))
            batch_labels = np.zeros((remaining_batch_size,5))

            for folder in range(remaining_batch_size):
                imgs = os.listdir(source_path+'/'+ t[folder + (num_batches*batch_size)].split(';')[0])
                for idx,item in enumerate(img_idx):
                    image = imageio.imread(source_path+'/'+ t[folder + (num_batches*batch_size)].strip().split(';')[0]+'/'+imgs[item]).astype(np.float32)

                    h, w, c = image.shape
                    image = resize(image, (height, width), anti_aliasing=True)

                    if augment:
                        if np.random.randn() > 0:
                            image = datagen.random_transform(image)

                    batch_data[folder,idx,:,:,0] = (image[...,0])/255
                    batch_data[folder,idx,:,:,1] = (image[...,1])/255
                    batch_data[folder,idx,:,:,2] = (image[...,2])/255

                batch_labels[folder, int(t[folder + (num_batches*batch_size)].strip().split(';')[2])] = 1

            yield batch_data, batch_labels