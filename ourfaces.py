import cv2
import os
import sys
import dlib
import numpy as np
from scipy import stats
from faceMorph import morphTriangle, calculateDelaunayTriangles
import traceback

FACE_MARGIN = 0.3

class OurFaces():
    def __init__(self):
        #https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
        self.faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    def alignFaces(self, faces):
        if len(faces) < 1:
            return None

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        w, h, rgb = faces[0].shape

        corners = [[0,0], [0, 499], [499, 0], [499,499], [round(499/2), 0], [499-1, round(499/2)], [round(499/2), 499-1], [0, round(499/2)]]

        print("Calculating average facial landmarks...")
        landmarksDict = {}
        landmarksList = []
        for index, face in enumerate(faces):
            dectectedFaces = detector(face, 1)

            if len(dectectedFaces) == 1:
                shape = predictor(face, dectectedFaces[0])
                landmarks = list(map(lambda i: [shape.part(i).x, shape.part(i).y], list(range(0, 68))))
                landmarksList.append(landmarks)
                landmarks.extend(corners)
                landmarksDict[index] = landmarks

        landmarksList = np.array(landmarksList)
        averageLandmarks = np.mean(landmarksList, axis=0).tolist()
        averageLandmarks = list(map(lambda i: [round(averageLandmarks[i][0]), round(averageLandmarks[i][1])], list(range(0, 68))))

        averageLandmarks.extend(corners)
        rect = (0, 0, 500, 500)
        delaunayTriangles = calculateDelaunayTriangles(rect, averageLandmarks)

        print("Morphing faces...")
        alignedFaces = []
        for index, landmarks in landmarksDict.items():
            face = np.float32(faces[index])
            imgMorph = np.zeros(face.shape, dtype = face.dtype)

            for triangle in delaunayTriangles:
                x, y, z = triangle
                t1 = [landmarks[x], landmarks[y], landmarks[z]]
                t = [averageLandmarks[x], averageLandmarks[y], averageLandmarks[z]]
                morphTriangle(face, imgMorph, t1, t)

            alignedFaces.append(np.uint8(imgMorph))

        return alignedFaces

    def findFacesInImage(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(gray, 1.3, 5)

        facesInImage = []

        for ( x, y, width, height) in faces:
            try:
                newX = int(round(x-width*FACE_MARGIN))
                newY = int(round(y-height*FACE_MARGIN))
                newWidth = int(round(width+2*width*FACE_MARGIN))
                newHeight = int(round(height+2*height*FACE_MARGIN))
                face = image[newY:newY+newHeight, newX:newX+newWidth]
                face = cv2.resize(face, (500, 500), interpolation = cv2.INTER_AREA)
                facesInImage.append(face)
            except:
                traceback.print_exc()

        return facesInImage

    def getModeFromImages(self, images, canvasSize):
        images = np.dstack(images)
        result = stats.mode(images, axis=2)
        return result.mode

    def stackImagesAboutCenter(self, mode, canvasSize, images):
        canvas = np.zeros((canvasSize, canvasSize, 3), np.float32())
        centeredImages = []

        for image in images:

            try:
                w, h, rgb = image.shape
                cw, ch = (int(round(w/2)), int(round(h/2)))
                ow, oh = (int(round(canvasSize/2))-cw, int(round(canvasSize/2))-ch)

                centeredImage = canvas.copy()
                centeredImage[ow:ow+w, oh:oh+h] = image

                if mode == "MODE":
                    centeredImage = cv2.cvtColor(centeredImage, cv2.COLOR_BGR2GRAY)
                centeredImages.append(centeredImage)
            except Exception as e:
                print('An error has occurred, skipping...')
                print(e)

        if mode == 'MEDIAN':
            result = np.median(centeredImages, axis=0)
        elif mode == "MEAN": 
            result = np.mean(centeredImages, axis=0)
        elif mode == "MODE":
            result = self.getModeFromImages(centeredImages, canvasSize)
            cv2.imshow('result', result)
        else:
            result = canvas
        return result