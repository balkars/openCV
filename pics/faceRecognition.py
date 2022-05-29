import cv2


# capture the devive
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

while True:
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if ret == False:
        continue

    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Video frame", frame)

    # wait for user input -q then stop the loop
    # ord gives us ASCII value
    # 0xFF gives us 8 1's

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
