import os
import tensorflow as tf
from tensorflow import keras
import cv2
import time
import numpy as np

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

cooldown = False

VIDEO_URL = "https://www.youtube.com/watch?v=jfKfPfyJRdk"
MODEL_PATH = "models/model-lstm-agument1.h5"
SWIPE_CONFIDENCE = 0.85
STOP_CONFIDENCE = 0.85
THUMBS_DOWN_CONFIDENCE = 0.85
THUMBS_UP_CONFIDENCE = 0.85
TEST = 0

print(TEST)

confidence_values = [SWIPE_CONFIDENCE, SWIPE_CONFIDENCE, STOP_CONFIDENCE, THUMBS_DOWN_CONFIDENCE, THUMBS_UP_CONFIDENCE]
classes = ["Swipe", "Swipe", "Stop", "Thumbs Down", "Thumbs Up"]

play_command = "document.querySelector('video').play();"
pause_command = "document.querySelector('video').pause();"
volume_up_command = """
const video = document.querySelector('video');
const event = new KeyboardEvent('keydown', {
    key: 'ArrowUp',
    code: 'ArrowUp',
    keyCode: 38,
    which: 38,
    bubbles: true
});
video.dispatchEvent(event);
"""
volume_down_command = """
const video = document.querySelector('video');
const event = new KeyboardEvent('keydown', {
    key: 'ArrowDown',
    code: 'ArrowDown',
    keyCode: 40,
    which: 40,
    bubbles: true
});
video.dispatchEvent(event);
"""

options = Options()
options.add_argument('--disable-dev-shm-usage')

if not TEST:
    print("DRIVER")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

commands = {
    "Swipe": play_command,
    "Stop": pause_command,
    "Thumbs Up": volume_up_command,
    "Thumbs Down": volume_down_command,
}


def handle_gesture(index, value):
    confidence_value = confidence_values[index]
    if value <= confidence_value:
        print("No gesture")

    elif value >= confidence_value:
        print(classes[index])
        print(value)
        return True


def execute_cooldown_callback():
    global cooldown
    time.sleep(1)
    cooldown = False

def main():
    global cooldown

    try:
        if not TEST:
            driver.get(VIDEO_URL)
    except:
        print ("Timed out waiting for page to load")
        time.sleep(5)

    model = keras.models.load_model(MODEL_PATH)
    model.summary()

    cap = cv2.VideoCapture(0)

    num_frames = 20
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (120, 120))
        frames.append(resized_frame)

        if len(frames) == num_frames:
            input_frames = np.array(frames)
            input_frames = np.expand_dims(input_frames, axis=0)
            input_frames = input_frames.astype('float32') / 255.0
            
            predictions = model.predict(input_frames)
            predicted_class = np.argmax(predictions)

            should_handle_gesture = handle_gesture(predicted_class, predictions[0][predicted_class])
            
            if should_handle_gesture and not cooldown:
                cooldown = True

                if not TEST:
                    driver.execute_script(commands[classes[predicted_class]])
                execute_cooldown_callback()

            frames = []
            

        cv2.imshow('WEBCAM', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if not TEST:
        driver.quit()

main()