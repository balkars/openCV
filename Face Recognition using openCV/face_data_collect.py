# this python script captures image from your webcam video stream
# extract all faces from the image frame ( using haarcascades)
# store the faces into numpy arrays

# steps :
# read and show video stream , capture images
# Detect Faces and show bounding box
# Flatten the largest face image and save it in a numpy array
# Repeat the above for multiple people for generating training data


# importing libraries

import cv2
