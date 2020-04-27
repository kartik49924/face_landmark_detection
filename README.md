# face_landmark_detection

This piece of code taken from [this](https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/) webpage (with slight changes) basically does 2 tasks-
1. It detects and localise face(s) in the input image.
2. It identifies and locates different landmarks (eye, nose, jawline, etc) for each face and store them in a .csv file for further usage.

Further, as per task given, it highlights 

Following image explains the location of each landmark point:

![landmark_image](https://www.pyimagesearch.com/wp-content/uploads/2017/04/facial_landmarks_68markup-768x619.jpg)

### How to use the code and generate results

Before running main script, following python libraries must be installed to their latest versions:
1. Numpy  (for general data flow)
2. Argparse  (for parsing arguments from command line)
3. imutils (for utility functions)
4. dlib (for utilizing trained detector and facial_features locator)
5. Open-cv (for image related utility functions)
6. Pandas (for exporting data)

Further, **

**facial_landmarks.py** script can be run from a command line as follows:

`python3 
