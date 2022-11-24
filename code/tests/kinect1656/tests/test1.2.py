# CITATION SOURCE: https://stackoverflow.com/questions/53905324/pykinect2-extract-depth-data-from-individule-pixel-kinectv2

import sys
sys.path.append("PyKinect2")

from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import numpy as np
import cv2

kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth)

import time
tot_framesC = 0
fpsC = 0
startC = time.time()

tot_framesD = 0
fpsD = 0
startD = time.time()


while True:
    # --- Getting frames and drawing
    if kinect.has_new_color_frame():
        tot_framesC += 1
        
        frame = kinect.get_last_color_frame()
        frame = frame.astype(np.uint8)
        frame = np.reshape(frame, (1080, 1920, 4))
        # instead of re-calculating fps too frequenty take more frames
            # we don't want instanteneous fps but rather average overall fps more accurately
        if tot_framesC == 5:
            end = time.time()
            tot_time = end - startC
            fpsC = round(tot_framesC/tot_time, 2)
            tot_framesC = 0
            startC = time.time()
        def click_eventC(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                scale_x = 1920/1280
                scale_y = 1080/720
                print(round(x*scale_x,0), round(y*scale_y,0))
        # this is rather the actual boundary rectangle (green) that matches to depth camera. here the reading values are correct. This has been approximately calculated via doing runtime tests
        cv2.rectangle(frame ,(270,0), (1790, 1080), (0, 255,0), 4) # do this before flipping
        # frame = cv2.flip(frame, 1)
        cv2.putText(frame, f'FPS: {fpsC}', (20,50), cv2.FONT_HERSHEY_COMPLEX, 1.3, (0,0,255))

        frame = cv2.resize(frame, (1280, 720))
        cv2.imshow('KINECT Color Stream', frame)
        cv2.setMouseCallback('KINECT Color Stream', click_eventC)
        output = None

    if kinect.has_new_depth_frame():
        tot_framesD += 1
        
        frame = kinect.get_last_depth_frame()
        frameD = kinect._depth_frame_data
        frame = frame.astype(np.uint8)
        frame = np.reshape(frame, (424, 512))
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        def click_eventD(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                print(x, y)
            if event == cv2.EVENT_RBUTTONDOWN:
                Pixel_Depth = frameD[((y * 512) + x)]
                print(Pixel_Depth)
        ##output = cv2.bilateralFilter(output, 1, 150, 75)

        # instead of re-calculating fps too frequenty take more frames
            # we don't want instanteneous fps but rather average overall fps more accurately
        if tot_framesD == 5:
            end = time.time()
            tot_time = end - startD
            fpsD = round(tot_framesD/tot_time, 2)
            tot_framesD = 0
            startD = time.time()

        # frame = cv2.flip(frame, 1)
        cv2.putText(frame, f'FPS: {fpsD}', (20,50), cv2.FONT_HERSHEY_COMPLEX, 1.3, (0,0,255))

        cv2.imshow('KINECT Depth Stream', frame)
        cv2.setMouseCallback('KINECT Depth Stream', click_eventD)
        output = None

    key = cv2.waitKey(1)
    if key == 27: break