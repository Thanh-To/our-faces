from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import base64

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import sys
import json

def index(request):
    print("yeet711")
    context = {}
    return render(request, 'myface/index.html', context)

@csrf_exempt
def testcall(request):
    imgdata = base64.b64decode(request.POST['img'].replace('data:image/png;base64,',''))

    filename = 'webScreenshot.jpg'

    with open(filename, 'wb') as f:
        f.write(imgdata)

    scoreDict = {}

    # Initiate SIFT detector
    sift = cv.xfeatures2d.SIFT_create()

    inputImg = cv.imread('webScreenshot.jpg',cv.IMREAD_GRAYSCALE)
    comparisonImagesPath = 'averages/'

    for subdir, dirs, files in os.walk(comparisonImagesPath):
        for file in files:

            comparisonImage = cv.imread(os.path.join(subdir,file),cv.IMREAD_GRAYSCALE) # trainImage

            print("Comparing input image to " + file)

            # find the keypoints and descriptors with SIFT
            kp1, des1 = sift.detectAndCompute(inputImg,None)
            kp2, des2 = sift.detectAndCompute(comparisonImage,None)

            # FLANN parameters
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks=50)   # or pass empty dictionary
            flann = cv.FlannBasedMatcher(index_params,search_params)
            matches = flann.knnMatch(des1,des2,k=2)

            # Need to draw only good matches, so create a mask

            goodMatchCounter = 0

            # ratio test as per Lowe's paper
            for i,(m,n) in enumerate(matches):
                if m.distance < 0.7*n.distance:
                    goodMatchCounter += 1

            faculty = file.split('_')[-1].split('.')[0]
            scoreDict[faculty] = goodMatchCounter

    return HttpResponse(json.dumps(scoreDict, indent = 4))

    
