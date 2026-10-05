# calibrate_brightness.py
import time
import numpy as np
import cv2
from picamera2 import Picamera2
import config

cam = Picamera2()
cfg = cam.create_preview_configuration(
    main={"size": (config.CAM_WIDTH, config.CAM_HEIGHT), "format": "RGB888"}
)
cam.configure(cfg)
cam.start()
time.sleep(2)

print("Mesure de luminosité — lance pendant le jour ET la nuit")
print("Ctrl+C pour arrêter\n")

while True:
    frame      = cam.capture_array()
    gray       = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    brightness = float(np.mean(gray))
    heure      = time.strftime("%H:%M:%S")
    print(f"{heure}  brightness = {brightness:.1f}")
    time.sleep(5)