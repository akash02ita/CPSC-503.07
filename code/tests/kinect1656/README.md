## Sources
- https://gist.github.com/joinAero/1f76844278f141cea8338d1118423648#file-camera-py
  - fails
- https://www.quora.com/How-can-I-execute-OpenCV-Python-Kinect-codes
- https://github.com/Kinect/PyKinect2
  - tried all possible ways: keeps failing
- https://stackoverflow.com/questions/51971493/how-to-print-a-kinect-frame-in-opencv-using-openni-bindings
  - do not konw what Redist has to do with it
- https://naman5.wordpress.com/2014/06/24/experimenting-with-kinect-using-opencv-python-and-open-kinect-libfreenect/
  - Windows support?
- https://github.com/OpenKinect/libfreenect
- current choice: https://github.com/Kinect/PyKinect2
  
## Notes
**cv2.Videocapture(number)** _does not work_ with Kinect v2. So something like PyKinect2 should be used to handle that. However currently all/most-of-the experiments failed.


## Status
Not currently working, after several attempts. Python 3.6 version seemed to work but was tested on 32 bit. 32 bit python does not support mediapipe pip install.

Python 3.6 64bit not tested yet.

Also seems like anaconda is not really needed but rather Python 3.6 is more than enough. Also replacing "PyKinect2\pykinect2\PyKinectRuntime.py" and "PyKinect2\pykinect2\PyKinectV2.py" in the 'virtual environment' location of these 2 files seems to fix most of the issues. For 64bit python versions some extra changes are needed. The github repository has some **issues** that talks about this.
