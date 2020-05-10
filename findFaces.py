import cv2
import os
import sys
import dlib
from alignFace import alignFace

FACE_MARGIN = 0.3

def findFacesInImage(image):
    #https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)

    count = 0
    alignedFaces = []

    for ( x, y, width, height) in faces:
        newX = int(round(x-width*FACE_MARGIN))
        newY = int(round(y-height*FACE_MARGIN))
        newWidth = int(round(width+2*width*FACE_MARGIN))
        newHeight = int(round(height+2*height*FACE_MARGIN))
        newImage = image[newY:newY+newHeight, newX:newX+newWidth]

        alignedFace = alignFace(newImage)
        if alignedFace is not None:
            alignedFaces.append(alignedFace)
            count += 1
            print(str(count) + ' faces found in image')
    
    return alignedFaces

argumentCount = len(sys.argv) - 1
if argumentCount != 2:
    print("usage: python findFaces.py [destinationFolderPath] [sourceFolderPath]")
else:
    destinationFolderPath = sys.argv[1]
    folderPath = sys.argv[2]
    os.mkdir(destinationFolderPath)
    filenum = 0
    for file in os.listdir(folderPath):
        if file.endswith(".jpg"):
            print(os.path.join(folderPath, file))

            img = cv2.imread(os.path.join(folderPath, file))
            faces = findFaces(img)
            for face in faces:
                filename = 'face' + str(filenum) + '.jpg'
                try:
                    cv2.imwrite(os.path.join(destinationFolderPath, filename), alignedFace)
                except:
                    print("An exception occurred with " + filename)
                print(filename)
                filenum += 1