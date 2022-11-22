# https://google.github.io/mediapipe/solutions/hands#python-solution-api
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

import pyrealsense2 as rs

# Create a context object. This object owns the handles to all connected realsense devices
pipeline = rs.pipeline()

# Configure streams
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
pipeline.start(config)

# For webcam input:
# cap = cv2.VideoCapture(0) # https://appdividend.com/2022/10/18/python-cv2-videocapture/
# cap = cv2.VideoCapture(1) # 2nd camera via usb
cap = cv2.VideoCapture(2) # 3rd camera via usb
# cap = cv2.VideoCapture("http://10.9.104.149:8080/video") # ip webcam: works pretty well
"""
 - maybe TRY TO SWITCH TO GPU first (CPU may be slow for real time video)
  - currently low fps and low cpu -> increase them
      - increasing cpu to high performance increases network video input (good) and then 
 - video size too large based on resolution: https://www.geeksforgeeks.org/how-to-change-video-resolution-in-opencv-in-python/
    - lower resolution? -> less cpu power and more fps?
    - resize video to fixed size or max size at most?

  SOURCES TO WATCH:
    - https://www.youtube.com/watch?v=8tdhm2rcB_c [60FPS CPU]

"""
with mp_hands.Hands(
    model_complexity=0,
    # model_complexity=1,
    # min_detection_confidence=0.5,
    # min_tracking_confidence=0.5) as hands:
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands: #0.5 seems to work best
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue
    
    # resize for lower resolution -> more fps? Nope still slow: seems like via webcam real time video open cv is slow.
    # image = cv2.resize(image, (720, 480))

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
    # if results.multi_hand_world_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
      # for hand_landmarks in results.multi_hand_world_landmarks:
        import time
        time.sleep(0.1)
        # image_width, image_height, _ = image.shape
        image_height, image_width, _ = image.shape
        print(
          f'Image shape {image_width} x {image_height}',
          f'Index finger tip coordinates: (',
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
        )
        x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width
        y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height
        frames = pipeline.wait_for_frames()
        depth = frames.get_depth_frame()
        if not depth:
          print("ERR: COULD NOT DETECT DEPTH!")
        elif 0 <= x <= image_width and 0 <= y <= image_height:
          dist = depth.get_distance(int(x), int(y))
          print("\tDepth of index finger:", dist)
        else:
          print("IGNORE out of bound (x,y)")
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    # cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()