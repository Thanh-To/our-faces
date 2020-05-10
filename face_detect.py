import numpy as np
import cv2
import matplotlib.pyplot as plt

#Loading the image to be tested
test_image = cv2.imread('data/images/Ex.jpg')

#Converting to grayscale
test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

haar_cascade_face = cv2.CascadeClassifier('data/xmls/haarcascade_frontalface_default.xml')

faces_rects = haar_cascade_face.detectMultiScale(test_image_gray, scaleFactor = 1.2, minNeighbors = 5);

print('Faces found: ', len(faces_rects))

for (x,y,w,h) in faces_rects:
     cv2.rectangle(test_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Displaying the grayscale image
cv2.imshow("test", test_image)

cv2.waitKey(0)
cv2.destroyAllWindows()