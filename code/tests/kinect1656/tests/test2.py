import sys

sys.path.append("PyKinect2")
from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
# import mapper for function to convert coordinates from color to depth space
sys.path.append("PyKinect2-Mapper-Functions")
import mapper
    

# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2

# import mediapipe for hand tracking
import mediapipe as mp

# import time for fps tracking
import time


mp_hands = mp.solutions.hands           # for processing image in tracking hand
mp_drawing = mp.solutions.drawing_utils # for drawing landmakrs on image
hands = mp_hands.Hands(                 # create hands object for processing image
    max_num_hands = 1,                  # at most one hand should be used
    model_complexity=0,min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

"""
    Concern: right now not really distinguishing between left or right hand
    But later can do that as well: example source:  https://toptechboy.com/distinguish-between-right-and-left-hands-in-mediapipe/
"""

# Streaming loop
tot_frames = 0
fps = 0
start = time.time()

# suppose 0 means invalid
depth_meters = 0

# use kinect camera in 2 modes: color and depth. Infrared mode is not needed
kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Color) # 8 | 1 = 1000 | 0001 = 1001

while True:
    temp_start = time.time()
    if kinect.has_new_depth_frame():
        # frame will be used to increment total frames (for fps tracking)
        tot_frames += 1
        color_frame = kinect.get_last_color_frame()
        depth_frame = kinect.get_last_depth_frame()
        

        color_image = color_frame.reshape((kinect.color_frame_desc.Height, kinect.color_frame_desc.Width, 4)).astype(np.uint8)
        depth_image = depth_frame.reshape((kinect.depth_frame_desc.Height, kinect.depth_frame_desc.Width)).astype(np.uint8)

        """
            Concern:
                when using mediapipe hands.process() the fps drops from 30 to 20. Not still sure why, despite lowering the resolution.
                The fps do not drop when using 'intel realsense d435i', where even over there mediapipe is used.
        """
        # color_image = cv2.resize(color_image, (600, 360)) # even a quick test via downscaling resolution does not help increase FPS
        image = color_image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # only in this format of coloring mediapipe is able to track and find hands
        results = hands.process(image) # apply hand tracking calculations: ONLY IF THIS LINE IS DISABLED and (cancelling if result.multi_hand_landmarks obviously) then speed goes back to 30FPS. Despite resizing and lowering resolution above, still low fps (around 20fps)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # restore color after being done processing image

        # if hand is tracked then landmarks will exist
        if results.multi_hand_landmarks:
            image_height, image_width, _ = image.shape
            # print(f'image_width, image_height, _ are {image_width}, {image_height}, {_}')
            for hand_landmarks in results.multi_hand_landmarks:
                x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width
                y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height

                # draw circle on index finger
                cv2.circle(image, (int(x),int(y)), radius=10, color=(0, 0, 255), thickness=-1)

                # map (x,y)-color-space to depth space
                # print(f"(x,y): {(x,y)}")
                depth_x, depth_y = -1, -1
                if 0 <= x <= 1920 and 0 <= y <= 1080:
                    # even if x and y are within the bounds, of depth space: if depth-z value is 0 then this seems to return depth_x=depth_y = 0
                    depth_x, depth_y = mapper.color_point_2_depth_point(kinect, _DepthSpacePoint, kinect._depth_frame_data, [int(x),int(y)])
                    cv2.putText(image, f"x,y: {(round(x,4),round(y,4))}", (450,800), cv2.FONT_HERSHEY_COMPLEX, 3, (0,255,0), 4)
                else:
                    print("IGNORE out of bound (x,y)")

                # print(f"depth_x, depth_y: {depth_x},{depth_y}")
                # depth_image_height, depth_image_width , _ = depth_image.shape
                depth_image_height, depth_image_width = depth_image.shape
                # depth_image_width, depth_image_height = depth_image.shape
                # print(f'depth_image_width, depth_image_height are {depth_image_width}, {depth_image_height}')
                
                if (depth_x | depth_y) and 0 <= depth_x <= depth_image_width and 0 <= depth_y <= depth_image_height: # bitwise or ensures checks whether depth_x = depth_y = 0
                    dist = mapper.depth_space_2_world_depth(depth_frame, depth_x, depth_y)
                    depth_meters = round(dist/1000, 4)
                    # print("\tDepth of index finger:", dist)
                    pass
                else:
                    # color_point_2_depth_point return (0,0) if out of bound
                    print("IGNORE out of bound (depth_x,depth_y) or invalid depth")
                    depth_meters = 0
                    pass

                # draw the hand landmarks on image
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        # instead of re-calculating fps too frequenty take more frames
            # we don't want instanteneous fps but rather average overall fps more accurately
        if tot_frames == 5:
            end = time.time()
            tot_time = end - start
            fps = round(tot_frames/tot_time, 2)
            tot_frames = 0
            start = time.time()

        # this is rather the actual boundary rectangle (green) that matches to depth camera. here the reading values are correct. This has been approximately calculated via doing runtime tests
        cv2.rectangle(image ,(270,0), (1790, 1080), (0, 255,0), 4) # do this before flipping
        # image = cv2.flip(image, 1)
        cv2.putText(image, f'FPS: {fps}', (20,150), cv2.FONT_HERSHEY_COMPLEX, 4, (0,0,255), 3)
        cv2.putText(image, f'DEPTH: {depth_meters}m', (800,150), cv2.FONT_HERSHEY_COMPLEX, 4, (255,0,0), 3)


        # since 1080*1920 (height, width) is too high, rescale resolution
        image = cv2.resize(image, (1280, 720)) # (width, height)
        cv2.imshow("image", image)
        cv2.imshow("depth", depth_image)
    # Press esc or 'q' to close the image window
    key = cv2.waitKey(5)
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()
        break

