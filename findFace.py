import cv2
import os
import sys

argumentCount = len(sys.argv) - 1
if argumentCount != 2:
    print("usage: python findFace.py [collectionName] [sourceFolderPath]")
else:
    collectionName = sys.argv[1]
    folderPath = sys.argv[2]
    os.mkdir(collectionName)

    for file in os.listdir(folderPath):
        filenum = 0
        if file.endswith(".jpg"):
            print(os.path.join(folderPath, file))

            img = cv2.imread(os.path.join(folderPath, file))

            #https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x,y,w,h) in faces:
                    x = x-70
                    y = y-70
                    w = w+140
                    h = h+140
                    roi_color = img[y:y+h, x:x+w]
                    filename = 'face' + str(filenum) + '.jpg'
                    cv2.imwrite(os.path.join(collectionName, filename), roi_color)
                    filenum += 1