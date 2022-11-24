# CITATION SOURCE: https://stackoverflow.com/questions/53905324/pykinect2-extract-depth-data-from-individule-pixel-kinectv2

import sys
sys.path.append("PyKinect2")

from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import numpy as np
import cv2

kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)

import time
tot_frames = 0
fps = 0
start = time.time()

while True:
    # --- Getting frames and drawing
    if kinect.has_new_color_frame():
        tot_frames += 1
        
        frame = kinect.get_last_color_frame()
        frame = frame.astype(np.uint8)
        frame = np.reshape(frame, (1080, 1920, 4))
        # instead of re-calculating fps too frequenty take more frames
            # we don't want instanteneous fps but rather average overall fps more accurately
        if tot_frames == 5:
            end = time.time()
            tot_time = end - start
            fps = round(tot_frames/tot_time, 2)
            tot_frames = 0
            start = time.time()

        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f'FPS: {fps}', (20,50), cv2.FONT_HERSHEY_COMPLEX, 1.3, (0,0,255))

        frame = cv2.resize(frame, (1280, 720))
        cv2.imshow('KINECT Video Stream', frame)
        output = None

    key = cv2.waitKey(1)
    if key == 27: break