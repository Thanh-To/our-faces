import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import sys
import json

scoreDict = {}

argumentCount = len(sys.argv) - 1
if argumentCount != 2:
    print("usage: python3 getMatchingScores.py [inputImagePath] [comparisonImagesPath]")
else:

	# Initiate SIFT detector
	sift = cv.xfeatures2d.SIFT_create()

	inputImg = cv.imread(sys.argv[1],cv.IMREAD_GRAYSCALE)
	comparisonImagesPath = sys.argv[2]

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

	print(json.dumps(scoreDict, indent = 4))