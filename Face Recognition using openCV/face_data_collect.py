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
import numpy as np

# Init camera
cap = cv2.VideoCapture(0)

# load the haarcascade file
# face detection

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

skip = 0
face_data = []
dataset_path = './data/'
file_name = input("Enter the name of the person : ")
while True:
    ret, frame = cap.read()

    if ret == False:
        continue

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)

    cv2.imshow("Frame", frame)

    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    faces = sorted(faces, key=lambda f: f[2]*f[3])

    for face in faces[-1:]:  # start from last /largest face
        x, y, w, h = face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

        # extract region of interest
        offset = 10
        face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section, (100, 100))

    cv2.imshow("frame", frame)

    # store every 10th face
    skip += 1
    if skip % 10 == 0:
        face_data.append(face_section)
        print(len(face_data))

    cv2.imshow("frame", frame)
    cv2.imshow("Face Section", face_section)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break
# convert our facelist into numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))
print(face_data.shape)


# save into filesystem

np.save(dataset_path+file_name+'.npy', face_data)
print('data successfully saved at ' + dataset_path+file_name+'.npy')

cap.release()
cv2.destroyAllWindows()
