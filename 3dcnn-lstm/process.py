import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

frames_cap = 30
classes = ['Swiping Left', 'Swiping Right', 'Sliding Two Fingers Left', 'Sliding Two Fingers Right', 'Doing other things']

def random_flip(image):
    choice = np.random.choice([1, 2, 3], p = (0.66, 0.16, 0.18))
    if choice == 1:
        return image
    elif choice == 2:
        return np.array(np.flipud(image))
    elif choice == 3:
        return np.array(np.fliplr(image))
    
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def unify_frames(path):
    frames = os.listdir(path)
    frames_count = len(frames)
    if frames_cap > frames_count:
        frames += [frames[-1]] * (frames_cap - frames_count)
    elif frames_cap < frames_count:
        frames = frames[0:frames_cap]
    return frames  

def resize_frame(frame):
    frame = img.imread(frame)
    frame = cv2.resize(frame, (64, 64))
    return frame

def extract_frames(csv, path):
    df = pd.read_csv(csv, index_col="VideoId")
    targets = (df.to_dict())["Label"]

    dirs = os.listdir(path)
    counter = 0
    y = []
    x = [] 

    for directory in dirs:
        frameset = []
        frames = unify_frames(path+directory)
        if len(frames) == frames_cap:
            for frame in frames:
                frame = resize_frame(path+directory+'/'+frame)
                frameset.append(rgb2gray(frame))
                if len(frameset) == 15:
                    x.append(frameset)
                    y.append(classes.index(targets[int(directory)]))
                    counter +=1
                    frameset = []

    return x, y

def extract_test_frames(path):
    test_frames = []
    dirs = os.listdir(path)

    counter = 0
    for directory in dirs:
        frameset = []
        frames = unify_frames(path+directory)
        if (len(frames) == frames_cap):
            for frame in frames:
                frame = resize_frame(path+directory+'/'+frame)
                frameset.append(rgb2gray(frame))
                if len(frameset) == 30:
                    test_frames.append(frameset)
                    counter += 1
                    frameset = []
    
    return test_frames

def normalize_data(np_data):
    scaled_images  = np_data.reshape(-1, 30, 64, 64, 1)
    return scaled_images

