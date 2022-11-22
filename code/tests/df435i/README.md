## Python version used
Python 3.9 is used. As of 17 Nov 2022, python 3.9 is the latest python version supported by `librealsense`.

## Setup virtual environment
1. install Python 3.9
  - for example for windows, as of 17 Nov 2022, Python 3.9.13 is one you can use
  - ADD TO PATH can be avoided, if another python version is installed
2. get the full path of python.exe where python 3.9 is installed
  - example: `C:\Users\akash\AppData\Local\Programs\Python\Python39\python.exe`
3. use pip from the same python version to create virtual environment
  - example: `C:\Users\akash\AppData\Local\Programs\Python\Python39\python.exe -m venv p39_env`
4. optional: inside environment folder create a `.gitignore` file
  - just add `*` to the .gitignore file. If you are using git the environment folder should be ignored.
5. now you can safely use python 3.9 in the virtual environment without conflicting with other python versions installed in your system, if there are 


## Source
- https://dev.intelrealsense.com/docs/python2