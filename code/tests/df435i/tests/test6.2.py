"""
    source: https://www.youtube.com/watch?v=iJCWdooOgfo
    experiment bitwise mask using trackbar

    note: RGB color space does not indeed to work as good as HSV
    not ideal
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
for g in device.sensors:
    if g.get_info(rs.camera_info.name) == 'RGB Camera':
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
lr, lg, lb = [0, 0, 0]
ur, ug, ub = [255, 255, 255]
limg = np.zeros((100,100,3), np.uint8)
himg = np.zeros((100,100,3), np.uint8)


# create trackbar list window
cv2.namedWindow("trackbar hsv", cv2.WINDOW_NORMAL)
cv2.resizeWindow("trackbar hsv", 800, 300)
cv2.createTrackbar("lr", "trackbar hsv", 0, 255, lambda _: None)  
cv2.createTrackbar("lg", "trackbar hsv", 0, 255, lambda _: None)  
cv2.createTrackbar("lb", "trackbar hsv", 0, 255, lambda _: None)  
cv2.createTrackbar("ur", "trackbar hsv", 255, 255, lambda _: None)
cv2.createTrackbar("ug", "trackbar hsv", 255, 255, lambda _: None)
cv2.createTrackbar("ub", "trackbar hsv", 255, 255, lambda _: None)


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

    while True:
        lr = cv2.getTrackbarPos("lr", "trackbar hsv")
        lg = cv2.getTrackbarPos("lg", "trackbar hsv")
        lb = cv2.getTrackbarPos("lb", "trackbar hsv")
        ur = cv2.getTrackbarPos("ur", "trackbar hsv")
        ug = cv2.getTrackbarPos("ug", "trackbar hsv")
        ub = cv2.getTrackbarPos("ub", "trackbar hsv")
        lsquare = np.full((200,200,3), [lr, lg, lb], dtype=np.uint8)
        usquare = np.full((200,200,3), [ur, ug, ub], dtype=np.uint8)
        lsquare = cv2.cvtColor(lsquare, cv2.COLOR_RGB2BGR)
        usquare = cv2.cvtColor(usquare, cv2.COLOR_RGB2BGR)
        cv2.imshow("lower bound color", usquare)
        cv2.imshow("upper bound color", lsquare)



        temp_start = time.time()
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames() # frames.get_depth_frame() is a 640x480 depth image

        depth_image = np.asanyarray(frames.get_depth_frame().get_data())
        depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        depth_image = cv2.flip(depth_image, 1)
        cv2.imshow("depth-before-processing-and-alignment", depth_image)

        # apply processing
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

        # experiment bitwise masks here
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # https://stackoverflow.com/questions/58624628/how-do-i-even-out-the-lighting-in-colored-images
        image = cv2.GaussianBlur(image, (5,5), 0)
        image = cv2.medianBlur(image,5)

        if lower_flag[0]:
            _,x,y = lower_flag
            print("lower", x, y, image[y, x])
            r,g,b = image[y, x]
            cv2.setTrackbarPos("lr", "trackbar hsv", r)
            cv2.setTrackbarPos("lg", "trackbar hsv", g)
            cv2.setTrackbarPos("lb", "trackbar hsv", b)
            lower_flag[0] = False
        if upper_flag[0]:
            _,x,y = upper_flag
            print("upper", x, y, image[y, x])
            r,g,b = image[y, x]
            cv2.setTrackbarPos("ur", "trackbar hsv", r)
            cv2.setTrackbarPos("ug", "trackbar hsv", g)
            cv2.setTrackbarPos("ub", "trackbar hsv", b)
            upper_flag[0] = False
        low_bound = (lr, lg, lb)
        upper_bound = (ur, ug, ub)

        mask = cv2.inRange(image, low_bound, upper_bound)
        image = cv2.bitwise_and(image, image, mask = mask)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

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
        cv2.setMouseCallback("image", set_bounds_on_click)
        
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