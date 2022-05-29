import cv2


# capture the devive
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if ret == False:
        continue

    cv2.imshow("Video frame", frame)
    cv2.imshow("gray frame", gray_frame)

    # wait for user input -q then stop the loop
    # ord gives us ASCII value
    # 0xFF gives us 8 1's

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
