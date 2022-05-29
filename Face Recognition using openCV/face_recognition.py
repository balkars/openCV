# recognise Faces using some classification algorithm- like KNN

# steps:
# 1. read the stream and extract faces out of it
# 2. load the training data(numpy arrays of all the persons)
# X : values are stored in the numpy arrays
# Y : values we need to assign for each person

# 3. use KNN to find the predicition of face
# 4. map the predicted id to name of the user
# 5. Display the predictions on the screen - boundary box and name of user


from multiprocessing.dummy import Namespace
import cv2
import numpy as np
import os

######KNN CODE #######


def distance(x1, x2):
    # Eucledian
    return np.sqrt(((x1-x2)**2).sum())


def knn(train, test, k=5):
    dist = []

    for i in range(train.shape[0]):

        # get the vector and label

        ix = train[i, :-1]
        iy = train[i, -1]

        # compute the distance
        d = distance(test, ix)
        dist.append([d, iy])

    # sort based on distance and get top k
    dk = sorted(dist, key=lambda x: x[0])[:k]

    # retreive only  labels
    labels = np.array(dk)[:, -1]

    # get frequencies of each label
    output = np.unique(labels, return_counts=True)

    # max freq and label
    index = np.argmax(output[1])
    return output[0][index]


# Init camera
cap = cv2.VideoCapture(0)

# Face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

skip = 0
dataset_path = './data/'
face_data = []
labels = []  # labels for given file

class_id = 0
names = {}  # maping between name and id

# data prep

for fx in os.listdir(dataset_path):
    if fx.endswith('npy'):

        # create a mapping between class id and name
        names[class_id] = fx[:-4]

        print("Loaded "+fx)
        data_item = np.load(dataset_path+fx)
        face_data.append(data_item)

        # create labels
        target = class_id*np.ones((data_item.shape[0], ))
        class_id += 1
        labels.append(target)


face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))

print(face_dataset.shape)
print(face_labels.shape)

trainset = np.concatenate((face_dataset, face_labels), axis=1)
print(trainset.shape)


# testing

while True:
    ret, frame = cap.read()
    if ret == False:
        continue

    faces = face_cascade.detectMultiScale(frame, 1.3, 5)

    for face in faces:
        x, y, w, h = face

        offset = 10
        # get region pf interest
        face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section, (100, 100))

        out = knn(trainset, face_section.flatten())

        # display on screen the name and rectangle around it
        pred_name = names[int(out)]
        cv2.putText(frame, pred_name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

    cv2.imshow("faces", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
