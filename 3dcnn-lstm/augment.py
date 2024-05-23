import tensorflow as tf
from keras import layers, preprocessing
from keras.models import Sequential
from keras.layers import Layer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from process import extract_frames, extract_single_sequence

# https://pyimagesearch.com/2021/06/28/data-augmentation-with-tf-data-and-tensorflow/

class VideoAugmentation(Layer):
  def __init__(self, **kwargs):
    super(VideoAugmentation, self).__init__(**kwargs)

  @tf.function
  def augment_frame(self, frame):
    augmented_frame = augment(frame)
    return augmented_frame
  
  def call(self, videos):
    return tf.map_fn(lambda video: tf.map_fn(self.augment_frame, video), videos)

def augment(input_data):
  augmentor = Sequential([
    layers.Rescaling(1./255),
    layers.RandomTranslation(
		height_factor=(-0.2, 0.2),
		width_factor=(-0.2, 0.2)),
    layers.RandomZoom(height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2)),
	  layers.RandomRotation(0.2),
    layers.RandomRotation(0.2),
    layers.GaussianNoise(0.1),
    layers.RandomContrast(0.2)
  ])

  return augmentor(input_data)

def plot_frames(image, img_type):
    # initialize a figure
    fig = plt.figure(figsize=(9, 9))
    fig.suptitle(img_type)
    # loop over the batch size
    for i in range(0, 8):
      # grab the image and label from the batch
      # create a subplot and plot the image and label
      ax = plt.subplot(2, 4, i + 1)
      plt.imshow(image[i])
      plt.axis("off")
    # show the plot
    plt.tight_layout()
    plt.show()