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

