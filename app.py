import cv2
import os
import sys
import dlib
from findFaces import findFacesInImage
from stackImages import stackImagesAboutCenter

DEPARTMENTS = ['CHBE','CIVIL','ECE','ENPH','GEOG','IGEN','MECH','MINE','MTRL']

argumentCount = len(sys.argv) - 1
if argumentCount != 3:
    print("usage: python app.py [mode (MEDIAN, MEAN)] [sourceFolderPath] 1")
else:
    mode = sys.argv[1]
    sourceFolderPath = sys.argv[2]

    for department in DEPARTMENTS:
        folderPath = os.path.join(sourceFolderPath, department)

        faces = []

        for file in os.listdir(folderPath):
            if file.endswith(".jpg"):
                print(os.path.join(folderPath, file))

                image = cv2.imread(os.path.join(folderPath, file))
                faces.extend(findFacesInImage(image))

        result = stackImagesAboutCenter(mode, 1000, faces)
        filePath = os.path.join(sourceFolderPath, department + '_' + '2014' + '_' + mode + '.jpg')
        cv2.imwrite(filePath, result)
