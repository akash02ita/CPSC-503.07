# https://google.github.io/mediapipe/solutions/hands#python-solution-api
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture(0) # https://appdividend.com/2022/10/18/python-cv2-videocapture/
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
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    # Flip the image horizontally for a selfie-view display.
    # cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()