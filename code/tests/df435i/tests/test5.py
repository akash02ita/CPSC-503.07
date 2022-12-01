"""
    from: test2.py

    TODO: apply 2 hand tracking in mediapipe
        - if left hand: start color segmentation
            1) filter only skin color till forearm: background shoudl be removed
            2) filter depth image as well: background should be removed
        - if right hand: that will be used for touching
            - TODO: distinguish between hover and touch

"""

# improvement of working_example1.py

# First import the library
import pyrealsense2 as rs
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
    max_num_hands = 2,                  # 2 hands will be used
    model_complexity=0,min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

"""
    Concern: right now not really distinguishing between left or right hand
    But later can do that as well: example source:  https://toptechboy.com/distinguish-between-right-and-left-hands-in-mediapipe/
"""

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

# without this doesn't work: invalid depth
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# without this doesn't work either: invalid color
if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)


# for post-processing and improving depth data
# processign follows pattern of 'https://github.com/IntelRealSense/librealsense/blob/jupyter/notebooks/depth_filters.
decimation = rs.decimation_filter()
decimation.set_option(rs.option.filter_magnitude, 4)

spatial = rs.spatial_filter()
spatial.set_option(rs.option.filter_magnitude, 5)
spatial.set_option(rs.option.filter_smooth_alpha, 1)
spatial.set_option(rs.option.filter_smooth_delta, 50)
spatial.set_option(rs.option.holes_fill, 3)

temporal = rs.temporal_filter()

hole_filling = rs.hole_filling_filter()

depth_to_disparity = rs.disparity_transform(True)
disparity_to_depth = rs.disparity_transform(False)


# Streaming loop
try:
    tot_frames = 0
    fps = 0
    start = time.time()

    # suppose 0 means invalid
    depth_meters = 0

    while True:
        temp_start = time.time()
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames() # frames.get_depth_frame() is a 640x480 depth image

        depth_image = np.asanyarray(frames.get_depth_frame().get_data())
        depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        depth_image = cv2.flip(depth_image, 1)
        cv2.imshow("depth-before-processing-and-alignment", depth_image)

        # https://github.com/IntelRealSense/librealsense/issues/2116: this does not help: get_distance seems out of range. Also it is best to post-process before aligning
        # https://github.com/IntelRealSense/librealsense/issues/2356: this helps more

        # Align the depth frame to color frame
        # aligned_frames = align.process(frames)
        # processign follows pattern of 'https://github.com/IntelRealSense/librealsense/blob/jupyter/notebooks/depth_filters.ipynb', putting everything together
        processed_frame = frames
        processed_frame = decimation.process(processed_frame).as_frameset()
        processed_frame = depth_to_disparity.process(processed_frame).as_frameset()
        processed_frame = spatial.process(processed_frame).as_frameset()
        processed_frame = temporal.process(processed_frame).as_frameset()
        processed_frame = disparity_to_depth.process(processed_frame).as_frameset()
        processed_frame = hole_filling.process(processed_frame).as_frameset()
        aligned_frames = align.process(processed_frame)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame:
            print("ERROR: aligned_depth_frame")
            pass
        if not color_frame:
            print("ERROR: color_frame")
            pass
        if not aligned_depth_frame or not color_frame:
            waste_of_time = time.time() - temp_start
            # ignore wasted time (otherwise might get inaccurate fps value)
            start += waste_of_time
            continue

        # frame will be used to increment total frames (for fps tracking)
        tot_frames += 1

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        image = color_image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # only in this format of coloring mediapipe is able to track and find hands
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # restore color after being done processing image
        if results.multi_hand_landmarks:
            hand_labels = []
            print("\t\tBegin results")
            print(results.multi_handedness) # contains the informatino we need
            # since image is not flipped: label: 'left' means user's right hand
            for idx, hand_handedness in enumerate(results.multi_handedness):
                # print(
                #     f'idx: {idx}\n'
                #     f'hand_handedness.classification[0].index: {hand_handedness.classification[0].index}\n' # int
                #     f'hand_handedness.classification[0].score: {hand_handedness.classification[0].score}\n' # float
                #     f'hand_handedness.classification[0].label: {hand_handedness.classification[0].label}\n' # str
                # )
                hand_labels.append(hand_handedness.classification[0].label)

            print("\t\tEnd results")
            # time.sleep(0.3)
            image_height, image_width, _ = image.shape
            # for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(results.multi_hand_landmarks)):
                print("i is ", i)
                hand_landmarks = results.multi_hand_landmarks[i]
                x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width
                y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height

                cv2.circle(image, (int(x),int(y)), radius=10, color=(0, 0, 255), thickness=-1)

                if hand_labels[i] == 'Right': # do not draw landmarks for users 'left' hand
                    continue

                # Validate that both frames are valid
                if not color_frame:
                    print("ERROR: NOT EXISTING VALID VALUES. color_frame unsuccessful")
                    pass
                if not aligned_depth_frame:
                    print("ERROR: NOT EXISTING VALID VALUES. aligned_depth_frame unsuccessful")
                    pass
                
                if color_frame and aligned_depth_frame and 0 <= x <= image_width and 0 <= y <= image_height:
                    dist = aligned_depth_frame.get_distance(int(x), int(y))
                    depth_meters = round(dist, 4)
                    # print("\tDepth of index finger:", dist)
                    pass
                elif not color_frame or not aligned_depth_frame:
                    pass
                else:
                    print("IGNORE out of bound (x,y)")
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

        image = cv2.flip(image, 1)
        cv2.putText(image, f'FPS: {fps}', (20,50), cv2.FONT_HERSHEY_COMPLEX, 1.3, (0,0,255))
        cv2.putText(image, f'DEPTH: {depth_meters}m', (300,50), cv2.FONT_HERSHEY_COMPLEX, 1.3, (255,0,0))
        cv2.imshow("image", image)
        
        depth_image = cv2.flip(depth_image, 1)
        depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        cv2.imshow("depth-after-processing-and-alignment", depth_image)
        # Press esc or 'q' to close the image window
        key = cv2.waitKey(5)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()