import tensorflow as tf
from keras import layers, preprocessing
from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from process import extract_frames, extract_single_sequence

# https://pyimagesearch.com/2021/06/28/data-augmentation-with-tf-data-and-tensorflow/

train_aug = Sequential([
  layers.Rescaling(scale=1.0/255),
	layers.RandomZoom(
		height_factor=(-0.05, -0.15),
		width_factor=(-0.05, -0.15)),
	layers.RandomRotation(0.15)
])

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

def augment_single_video(vid):
  augmented_frames = []

  for frame in vid:
    augmented_frames.append(train_aug(frame))
  
  return augmented_frames

def get_augmented_set():
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

def main():
  video = np.array([extract_single_sequence("./trainset/", "20")])
  video = video.reshape(15, 64, 64, 3)

  augmented_video = augment_single_video(video)
  plot_frames(augmented_video, "Augmented")

  # scaler = StandardScaler()
  # scaled_video  = scaler.fit_transform(video.reshape(-1, 15*64*64))
  # scaled_video  = scaled_video.reshape(-1, 15, 64, 64, 1)

main()