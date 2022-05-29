import cv2
img = cv2.imread('pics/sample.jpg')
cv2.imshow('dog pic', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
