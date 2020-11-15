import cv2
import os
import sys
import dlib
from ourfaces import OurFaces

FACE_MARGIN = 0.3

argumentCount = len(sys.argv) - 1
if argumentCount != 2:
    print("usage: python app.py [mode (MEDIAN, MEAN)] [sourceFolderPath]")
else:
    mode = sys.argv[1]
    folderPath = sys.argv[2]

    ourFaces = OurFaces()
    faces = []

    for file in os.listdir(folderPath):
        if file.endswith((".jpg", ".JPG")):
            print(os.path.join(folderPath, file))
            image = cv2.imread(os.path.join(folderPath, file))
            faces.extend(ourFaces.findFacesInImage(image))

    faces = ourFaces.alignFaces(faces)
    print("Stacking faces...")
    result = ourFaces.stackImagesAboutCenter(mode, 1000, faces)
    filePath = os.path.join(folderPath, mode + '.jpg')
    cv2.imwrite(filePath, result)
