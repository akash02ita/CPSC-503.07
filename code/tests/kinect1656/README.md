## Sources
- current choice: https://github.com/Kinect/PyKinect2
- another choice: https://github.com/KonstantinosAng/PyKinect2-Mapper-Functions
  - uses pykinect2 library to apply coordinate alignment.
  
## Notes
**cv2.Videocapture(number)** _does not work_ with Kinect v2. So something like PyKinect2 should be used to handle that. Successfully working with PyKinect2.

Regarding OpenKinect, which is libfreenect (for Kinect) and libfreenect2 (for Kinect v2) there are python wrappers such as **freenect2** and **pylibfreenect2**. However those do not seem to work with Python 3.7 64 bit (which is minimum requirement for **mediapipe**). PyKinect2 seems the best choice so far.


## Status
Working with **PyKinect2**. Using Python 3.7 64 bit at the moment.

So this is the procedure:
- do not use `pip install pykinect2`. Seems outdated or not working with 64bit Python.
- rather just use/import `PyKinectRuntime.py` and `PyKinectV2.py` from the __pykinect2__ github repository. That works with Python 3.7 64bit (tested).


## Examples
- `test/test2.py` is a working example with hand tracking. The current issue is that the FPS drops from 30 to 20 when using mediapipe **hands.process()**. The issue persists despite lowering resolution of numpy array. However this issue was not encountered with __intel realsense d453i__: FPS stays between 22-30, even after post-processing.
  - tests performed under **Ryzen 5500U** and **12gb RAM**.
  - in d345i, using Python 3.9 64bit, whereas in kinect1656 using Python 3.7 64bit. Not sure if (1) mediapipe performance depends on different python versions and/or (2) mediapipe version is different with Python 3.7.