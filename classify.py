import cv2
import os
import sys
from deepface import DeepFace
from deepface.extendedmodels import Gender, Race

FACE_MARGIN = 0.3

argumentCount = len(sys.argv) - 1
if argumentCount != 1:
    print("usage: python classify.py [sourceFolderPath]")
else:
    sourceFolderPath = sys.argv[1]

    #https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    models = {}
    models["gender"] = Gender.loadModel()
    models["race"] = Race.loadModel()

    folderPath = sourceFolderPath

    for file in os.listdir(folderPath):
        if file.endswith((".jpg", ".JPG")):

            print(os.path.join(folderPath, file))
            image = cv2.imread(os.path.join(folderPath, file))

            # resize image
            height = 1000
            width = int(image.shape[1] * (height / image.shape[0]))
            dim = (width, height)
            image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
            image = cv2.copyMakeBorder(image, 200, 200, 200, 200, cv2.BORDER_CONSTANT, None, [0, 0, 0])

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.3, 5)

            for ( x, y, width, height) in faces:
                try:
                    newX = int(round(x-width*FACE_MARGIN))
                    newY = int(round(y-height*FACE_MARGIN))
                    newWidth = int(round(width+2*width*FACE_MARGIN))
                    newHeight = int(round(height+2*height*FACE_MARGIN))
                    image = image[newY:newY+newHeight, newX:newX+newWidth]

                    demography = DeepFace.analyze(image, actions = ['gender', 'race'], models=models)
                    genderDir = os.path.join(folderPath, demography["gender"])
                    raceDir = os.path.join(folderPath, demography["dominant_race"])
                    print(os.path.join(genderDir, file))
                    print(os.path.join(raceDir, file))
                    if not os.path.exists(genderDir):
                        os.makedirs(genderDir)
                    if not os.path.exists(raceDir):
                        os.makedirs(raceDir)
                    cv2.imwrite(os.path.join(folderPath, demography["gender"], file), image)
                    cv2.imwrite(os.path.join(folderPath, demography["dominant_race"], file), image)
                except Exception as e:
                    print(e)
                    print("skipping...")