# USAGE
# python facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg 

# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import pandas as pd
import numpy.polynomial.polynomial as poly


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image
rects = detector(gray, 1)

# loop over the face detections
faces=[]
for (i, rect) in enumerate(rects):
	# determine the facial landmarks for the face region, then
	# convert the facial landmark (x, y)-coordinates to a NumPy
	# array
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)

	# convert dlib's rectangle to a OpenCV-style bounding box
	# [i.e., (x, y, w, h)], then draw the face bounding box
	(x, y, w, h) = face_utils.rect_to_bb(rect)
	#cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

	# show the face number
	cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

	# loop over the (x, y)-coordinates for the facial landmarks
	# and draw them on the image
	#for (x, y) in shape:
	#	cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
	x=shape[:17,0].reshape(17,)
	y=shape[:17,1].reshape(17,)
	coefs = poly.polyfit(x, y, 20)
	ffit = poly.Polynomial(coefs)
	y_new=ffit(x)
	for i in range(1,17): 
		cv2.line(image, (x[i-1],int(y_new[i-1])), (x[i],int(y_new[i])), (0,0,255),2 )
	faces.append(shape)

faces=tuple(faces)
face_count=len(faces)
column_names=[]
data=np.hstack(faces)
for i in range(face_count):
	column_names.append('Face_'+str(i+1)+'x')
	column_names.append('Face_'+str(i+1)+'y')
shape_data=pd.DataFrame(data,columns=column_names)
shape_data.to_csv("~/Downloads/face_landmarks.csv",index=False)
# show the output image with the face detections + facial landmarks
cv2.imshow("Output", image)
cv2.waitKey(0)
