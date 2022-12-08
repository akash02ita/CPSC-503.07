"""
    from: test7.1.py
    source: https://docs.opencv.org/4.x/d5/d0f/tutorial_py_gradients.html
    source: https://docs.opencv.org/3.4/d2/d2c/tutorial_sobel_derivatives.html
    source: https://www.researchgate.net/publication/291000732_Comparison_of_Edge_Detection_Algorithms_for_Automated_Radiographic_Measurement_of_the_Carrying_Angle
    by reading the source seems that sobel filter should be good enough
"""

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


# lower and upper (h)ue (s)aturation and (v)alue
lh, ls, lv = [0, 0, 0]
uh, us, uv = [255, 255, 255]
limg = np.zeros((100,100,3), np.uint8)
himg = np.zeros((100,100,3), np.uint8)


# create trackbar list window
cv2.namedWindow("trackbar hsv", cv2.WINDOW_NORMAL)
cv2.resizeWindow("trackbar hsv", 800, 300)
cv2.createTrackbar("lh", "trackbar hsv", 0, 179, lambda _: None) # in cv2 hue range is [0, 179]
cv2.createTrackbar("ls", "trackbar hsv", 0, 255, lambda _: None) # in cv2 saturation range is [0, 255]
cv2.createTrackbar("lv", "trackbar hsv", 0, 255, lambda _: None) # in cv2 hue value is [0, 255]
cv2.createTrackbar("uh", "trackbar hsv", 179, 179, lambda _: None) # in cv2 hue range is [0, 179]
cv2.createTrackbar("us", "trackbar hsv", 255, 255, lambda _: None) # in cv2 saturation range is [0, 255]
cv2.createTrackbar("uv", "trackbar hsv", 255, 255, lambda _: None) # in cv2 hue value is [0, 255]

# https://pyimagesearch.com/2014/08/18/skin-detection-step-step-example-using-python-opencv/ (0, 48, 80 and 20, 255, 255 mask works pretty well when mask is applied gaussian blur)
cv2.setTrackbarPos("lh", "trackbar hsv", 0)
cv2.setTrackbarPos("ls", "trackbar hsv", 48)
cv2.setTrackbarPos("lv", "trackbar hsv", 80)
cv2.setTrackbarPos("uh", "trackbar hsv", 20)
cv2.setTrackbarPos("us", "trackbar hsv", 255)
cv2.setTrackbarPos("uv", "trackbar hsv", 255)
# cv2.setTrackbarPos("lh", "trackbar hsv", 0)
# cv2.setTrackbarPos("ls", "trackbar hsv", 59)
# cv2.setTrackbarPos("lv", "trackbar hsv", 49)
# cv2.setTrackbarPos("uh", "trackbar hsv", 20)
# cv2.setTrackbarPos("us", "trackbar hsv", 255)
# cv2.setTrackbarPos("uv", "trackbar hsv", 255)


lower_flag = [False, None, None]
upper_flag = [False, None, None]

def set_bounds_on_click(event,x,y,flags,param):
    global lower_flag, upper_flag
    if event == cv2.EVENT_LBUTTONDBLCLK:
        lower_flag = [True, x, y]
    if event == cv2.EVENT_RBUTTONDBLCLK:
        upper_flag = [True, x, y]
    pass

# Streaming loop
try:
    tot_frames = 0
    fps = 0
    start = time.time()

    # suppose 0 means invalid
    depth_meters = 0

    left_hand_on = False
    right_hand_on = False

    left_wrist_x = left_wrist_y = None
    right_index_finger_tip_x = right_index_finger_tip_y = None




    while True:
        lh = cv2.getTrackbarPos("lh", "trackbar hsv")
        ls = cv2.getTrackbarPos("ls", "trackbar hsv")
        lv = cv2.getTrackbarPos("lv", "trackbar hsv")
        uh = cv2.getTrackbarPos("uh", "trackbar hsv")
        us = cv2.getTrackbarPos("us", "trackbar hsv")
        uv = cv2.getTrackbarPos("uv", "trackbar hsv")
        lsquare = np.full((200,200,3), [lh, ls, lv], dtype=np.uint8)
        usquare = np.full((200,200,3), [uh, us, uv], dtype=np.uint8)
        lsquare = cv2.cvtColor(lsquare, cv2.COLOR_HSV2BGR)
        usquare = cv2.cvtColor(usquare, cv2.COLOR_HSV2BGR)
        cv2.imshow("lower bound color", lsquare)
        cv2.imshow("upper bound color", usquare)


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
        derivative_depth_image = None

        image = color_image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # only in this format of coloring mediapipe is able to track and find hands
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # restore color after being done processing image
        if results.multi_hand_landmarks:
            # since image is not flipped: label: 'left' means user's right hand
            image_height, image_width, _ = image.shape
            for idx, hand_handedness in enumerate(results.multi_handedness):
                hand_landmarks = results.multi_hand_landmarks[idx]

                if hand_handedness.classification[0].label == 'Right': # do not draw landmarks for users 'left' hand
                    x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width
                    y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height
                    
                    left_hand_on = True
                    left_wrist_x, left_wrist_y = x, y

                    # apply color segmentation
                    # experiment bitwise masks here
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

                    # https://stackoverflow.com/questions/58624628/how-do-i-even-out-the-lighting-in-colored-images
                    image = cv2.GaussianBlur(image, (5,5), 0)
                    image = cv2.medianBlur(image,5)

                    if lower_flag[0]:
                        _,x,y = lower_flag
                        print("lower", x, y, image[y, x])
                        h,s,v = image[y, x]
                        cv2.setTrackbarPos("lh", "trackbar hsv", h)
                        cv2.setTrackbarPos("ls", "trackbar hsv", s)
                        cv2.setTrackbarPos("lv", "trackbar hsv", v)
                        lower_flag[0] = False
                    if upper_flag[0]:
                        _,x,y = upper_flag
                        print("upper", x, y, image[y, x])
                        h,s,v = image[y, x]
                        cv2.setTrackbarPos("uh", "trackbar hsv", h)
                        cv2.setTrackbarPos("us", "trackbar hsv", s)
                        cv2.setTrackbarPos("uv", "trackbar hsv", v)
                        upper_flag[0] = False
                    low_bound = (lh, ls, lv)
                    upper_bound = (uh, us, uv)
                    # source: https://stackoverflow.com/questions/8753833/exact-skin-color-hsv-range
                    # 0, 58, 50 and 0, 255, 255 works very well --> enhanced to 0,59,49, and 20, 255, 255
                    # 0, 10, 60 and 20, 150, 255 also works
                    # ^^ note that they work in good lighting conditions: MS 680 all lights around works properly
                    mask = cv2.inRange(image, low_bound, upper_bound)
                    mask = cv2.GaussianBlur(mask, (3,3), 0)
                    image = cv2.bitwise_and(image, image, mask = mask)
                    image = cv2.medianBlur(image, 5)
                    
                    # do these next todo steps below this for loop  
                    # STEP1: remove corresponding background in depth
                    # efficient approach: use numpy methods to have fast process
                    condition = image != 0 # select NON-BLACK pixels
                    condition = condition.any(axis = 2) # so if ANY of the 3 bgr pixels entries are != 0, then pixel is NON-BLACK
                    depth_image = np.where(condition, depth_image, 0)
                    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)


                    # only if both left and right hands are on frame then do the next 2 steps
                    # STEP2 derivative of depth image 
                    # gray = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY) # gives error: why?
                    # gray = depth_image
                    gray = cv2.GaussianBlur(depth_image, (3, 3), 0)
                    ddepth, scale, delta = cv2.CV_16S, 1, 0
                    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
                    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
                    abs_grad_x = cv2.convertScaleAbs(grad_x)
                    abs_grad_y = cv2.convertScaleAbs(grad_y)
                    derivative_depth_image = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
                    if (derivative_depth_image is None):
                        print("OPS!!!!!!!!!!!")
                    # depth_image = grad
                    cv2.imshow("depth derivative", cv2.flip(derivative_depth_image, 1))
                    # depth_image = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)
                    



                    # after color segmentation this part is optional
                    cv2.circle(image, (int(x),int(y)), radius=10, color=(0, 0, 255), thickness=-1)
                    continue


                
                # otherwise right hand is on

                # Validate that both frames are valid
                if not color_frame:
                    print("ERROR: NOT EXISTING VALID VALUES. color_frame unsuccessful")
                    pass
                if not aligned_depth_frame:
                    print("ERROR: NOT EXISTING VALID VALUES. aligned_depth_frame unsuccessful")
                    pass
                
                x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width
                y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height
                

                if color_frame and aligned_depth_frame and 0 <= x <= image_width and 0 <= y <= image_height:
                    # dist = aligned_depth_frame.get_distance(int(x), int(y)) # this works
                    dist = depth_image[int(y),int(x)]/1000 # this works but now using depth_image, since it is preprocessed
                    depth_meters = round(dist, 4)
                    # print("\tDepth of index finger:", dist)
                    right_hand_on = True
                    right_index_finger_tip_x, right_index_finger_tip_y = x, y
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



        # STEP3 flood fill approach to distinguish hover or touch
        if left_hand_on and right_hand_on and (derivative_depth_image is not None):
            # print("both hands on")

            
            x, y = int(right_index_finger_tip_x), int(right_index_finger_tip_y)
            dx, dy = 10, 10
            square = depth_image[y-dy:y+dy, x-dx:x+dx]
            der_square = derivative_depth_image[y-dy:y+dy, x-dx:x+dx]                

            if ((square == 0).all() or (square == 0).sum() >= int(dx*dy*0.95)): # PASS 1 check if finger is away from forearm
                image = cv2.flip(image, 1)
                # print("Finger away") 
                cv2.putText(image, "away", (50, 400), cv2.FONT_HERSHEY_COMPLEX, 1.3, (255,0,255), 2)
                image = cv2.flip(image, 1)
            elif ((der_square == 0).all() or (der_square == 0).sum() >= int(dx*dy*0.98)): # PASS 2 check if finger is touching
                image = cv2.flip(image, 1)
                # print("Finger touching") 
                cv2.putText(image, "touch", (50, 400), cv2.FONT_HERSHEY_COMPLEX, 1.3, (255,0,255), 2)
                image = cv2.flip(image, 1)
            else:
                image = cv2.flip(image, 1)
                # print("Finger Hovering") 
                cv2.putText(image, "hover", (50, 400), cv2.FONT_HERSHEY_COMPLEX, 1.3, (255,0,255), 2)
                image = cv2.flip(image, 1)

            left_hand_on = right_hand_on = False

        # instead of re-calculating fps too frequenty take more frames
            # we don't want instanteneous fps but rather average overall fps more accurately
        if tot_frames == 5:
            end = time.time()
            tot_time = end - start
            fps = round(tot_frames/tot_time, 2)
            tot_frames = 0
            start = time.time()

        image = cv2.flip(image, 1)
        cv2.putText(image, f'FPS: {fps}', (20,50), cv2.FONT_HERSHEY_COMPLEX, 1.3, (0,0,255), 3)
        cv2.putText(image, f'DEPTH: {depth_meters}m', (300,50), cv2.FONT_HERSHEY_COMPLEX, 1.3, (255,255,0), 3)
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