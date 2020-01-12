import cv2
import os
import sys

argumentCount = len(sys.argv) - 1
if argumentCount != 2:
    print("usage: python findFace.py [destinationFolderPath] [sourceFolderPath]")
else:
    destinationFolderPath = sys.argv[1]
    folderPath = sys.argv[2]
    os.mkdir(destinationFolderPath)
    filenum = 0
    for file in os.listdir(folderPath):
        if file.endswith(".jpg"):
            print(os.path.join(folderPath, file))

            img = cv2.imread(os.path.join(folderPath, file))

            #https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
            faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.3, 5)

            for (x,y,width,height) in faces:
                    newX = max(0,int(round(x-width*0.3)))
                    newY = max(0,int(round(y-height*0.3)))
                    newWidth = int(round(width+2*width*0.3))
                    newHeight = int(round(height+2*height*0.3))
                    newImg = img[newY:newY+newHeight, newX:newX+newWidth]
                    filename = 'face' + str(filenum) + '.jpg'
                    cv2.imwrite(os.path.join(destinationFolderPath, filename), newImg)
                    filenum += 1