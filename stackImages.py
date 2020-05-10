import cv2
import numpy as np
import os
import sys
from scipy import stats

def imagesMode(images, canvasSize):
    images = np.dstack(images)
    result = stats.mode(images, axis=2)
    return result.mode

def stackImagesAboutCenter(mode, canvasSize, images):
    canvas = np.zeros((canvasSize, canvasSize, 3), np.float32())
    centeredImages = []

    for image in images:
        try:
            w, h, rgb = image.shape
            cw, ch = (int(round(w/2)), int(round(h/2)))
            ow, oh = (int(round(canvasSize/2))-cw, int(round(canvasSize/2))-ch)

            centeredImage = canvas.copy()
            centeredImage[oh:oh+h, ow:ow+w] = image

            if mode == "MODE":
                centeredImage = cv2.cvtColor(centeredImage, cv2.COLOR_BGR2GRAY)
            centeredImages.append(centeredImage)
        except:
            print('An error has occurred, skipping...')

    if mode == 'MEDIAN':
        result = np.median(centeredImages, axis=0)
    elif mode == "MEAN": 
        result = np.mean(centeredImages, axis=0)
    elif mode == "MODE":
        result = imagesMode(centeredImages, canvasSize)
        cv2.imshow('result', result)
    else:
        result = canvas
    return result


argumentCount = len(sys.argv) - 1
if argumentCount != 4:
    print("usage: python stackImages.py [mode (MEDIAN, MEAN)] [squareCanvasSize] [destinationFilePath] [sourceFolderPath]")
else:
    mode = sys.argv[1]
    canvasSize = int(sys.argv[2])
    destinationFilePath = sys.argv[3]
    folderPath = sys.argv[4]

    images = []

    for file in os.listdir(folderPath):
        if file.endswith(".jpg"):
            print(os.path.join(folderPath, file))
            image = cv2.imread(os.path.join(folderPath, file))
            images.append(image)

    result = stackImagesAboutCenter(mode, canvasSize, images)
    cv2.imwrite(destinationFilePath, result)


                
