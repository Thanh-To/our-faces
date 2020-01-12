import cv2
import os
import sys
import dlib
from findFaces import findFacesInImage
from stackImages import stackImagesAboutCenter

argumentCount = len(sys.argv) - 1
if argumentCount != 3:
    print("usage: python app.py [mode (MEDIAN, MEAN)] [DEPARTMENT (eg. ECE, MECH)] [sourceFolderPath]")
else:
    mode = sys.argv[1]
    department = sys.argv[2]
    folderPath = sys.argv[3]
    sourceFolderPath = os.path.join(folderPath, department)

    faces = []

    for file in os.listdir(sourceFolderPath):
        if file.endswith(".jpg"):
            print(os.path.join(sourceFolderPath, file))

            image = cv2.imread(os.path.join(sourceFolderPath, file))
            faces.extend(findFacesInImage(image))
    
    result = stackImagesAboutCenter(mode, 1000, faces)
    filePath = os.path.join(folderPath, department + '_' + mode + '.jpg')
    cv2.imwrite(filePath, result)
