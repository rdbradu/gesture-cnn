import time
import cv2
import keras
import numpy as np

from augment import VideoAugmentation

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

model = keras.models.load_model("3dcnn_lstm.h5", custom_objects={"VideoAugmentation": VideoAugmentation})

options = Options()
# options.add_argument('--headless')
# options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

driver.get("https://www.youtube.com/watch?v=s1tkN8scaQY")

cap = cv2.VideoCapture(0)

play_command = "document.querySelector('video').play();"
pause_command = "document.querySelector('video').pause();"
volume_up_command = "document.querySelector('video').volume = Math.min(document.querySelector('video').volume + 0.1, 1);"
volume_down_command = "document.querySelector('video').volume = Math.max(document.querySelector('video').volume - 0.1, 0);"

commands = {
    "play": play_command,
    "pause": pause_command,
    "volume_up": volume_up_command,
    "volume_down": volume_down_command,
}

classes = ['pause', 'play', 'volume_up', 'volume_down', None]

def get_gesture():
    frames = []

    ret, frame = cap.read()

    while len(frames) < 30:
        frame = cv2.resize(frame, (112, 112))
        frames.append(frame)

    return np.array(frames, dtype=np.float32)

try:
    element_present = EC.presence_of_element_located((By.TAG_NAME, 'video'))
    WebDriverWait(driver, 5).until(element_present)
except TimeoutException:
    print ("Timed out waiting for page to load")

while True:
    gesture = get_gesture()
    gesture = gesture.reshape(-1, 30, 112, 112, 3)
    prediction = model.predict(gesture)
    action = classes[np.argmax(prediction)]

    if action:
        driver.execute_script(commands[action])
    
    time.sleep(1)
    # print(driver.title)
