import cv2
img = cv2.imread('pics/sample.jpg')
cv2.imshow('dog pic', img)


# opencv always read RGB images as BGR images , if we use imshow of opencv instead of matplotlib we will get
# BGR as BGR

# earlier --> read as BGR and displayed as RGB(matplotlib)
# Now     --> read as BGR and displayed as BGR
cv2.waitKey(0)
cv2.destroyAllWindows()
