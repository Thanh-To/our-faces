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
import io

from google.cloud import vision
from google.cloud.vision import types

from PIL import Image, ImageDraw

def index(request):
    context = {}
    return render(request, 'myface/index.html', context)

@csrf_exempt
def testcall(request):

    # Save the image you got from the request
    imgdata = base64.b64decode(request.POST['img'].replace('data:image/png;base64,',''))

    filename = 'webScreenshot.jpg'

    with open(filename, 'wb') as f:
        f.write(imgdata)

    client = vision.ImageAnnotatorClient()

    with open(filename, 'rb') as image:
        faces = detect_face(image, 1)
        print('Found {} face{}'.format(
            len(faces), '' if len(faces) == 1 else 's'))

        print('Writing to file {}'.format(filename))
        # Reset the file pointer, so we can read the file again
        image.seek(0)
        crop_face(image, faces, filename)

    scoreDict = {}

    # Initiate SIFT detector
    sift = cv.xfeatures2d.SIFT_create()

    inputImg = cv.imread('webScreenshot.jpg',cv.IMREAD_GRAYSCALE)
    comparisonImagesPath = 'averages/'

    for subdir, dirs, files in os.walk(comparisonImagesPath):
        for file in files:
            if file.endswith('.jpg'):
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

    sorted_score = {k: v for k, v in sorted(scoreDict.items(), key=lambda x: x[1], reverse=True)}
    print(sorted_score)

    context = {'scores':list(sorted_score.keys())}
    return render(request, "myface/result.html", context)

def detect_face(face_file, max_results=4):
    """Uses the Vision API to detect faces in the given file.

    Args:
        face_file: A file-like object containing an image with faces.

    Returns:
        An array of Face objects with information about the picture.
    """
    client = vision.ImageAnnotatorClient()

    content = face_file.read()
    image = types.Image(content=content)

    return client.face_detection(
        image=image, max_results=max_results).face_annotations

def crop_face(image, faces, output_filename):
    """Draws a polygon around the faces, then saves to output_filename.

    Args:
      image: a file containing the image with the faces.
      faces: a list of faces found in the file. This should be in the format
          returned by the Vision API.
      output_filename: the name of the image file to be created, where the
          faces have polygons drawn around them.
    """
    im = Image.open(image)
    rgb_im = im.convert('RGB')
    draw = ImageDraw.Draw(rgb_im)

    # Sepecify the font-family and the font-size
    for face in faces:

        left = face.bounding_poly.vertices[0].x
        top = face.bounding_poly.vertices[0].y
        right = face.bounding_poly.vertices[2].x
        bottom = face.bounding_poly.vertices[2].y

        print("LEFT: " + str(left))
        print("RIGHT: " + str(right))
        print("TOP: " + str(top))
        print("BOTTOM: " + str(bottom))

        cropped = rgb_im.crop((left,top,right,bottom))
        cropped.save(output_filename)