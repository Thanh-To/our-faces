import cv2
import os
import sys
import dlib
from utils import extract_left_eye_center, extract_right_eye_center, get_rotation_matrix

def alignFace(image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    height, width = image.shape[:2]
    s_height, s_width = height, width
    image = cv2.resize(image, (s_width, s_height))

    dets = detector(image, 1)

    if len(dets) == 1:
        shape = predictor(image, dets[0])
        left_eye = extract_left_eye_center(shape)
        right_eye = extract_right_eye_center(shape)

        M = get_rotation_matrix(left_eye, right_eye)
        rotated = cv2.warpAffine(image, M, (s_width, s_height), flags=cv2.INTER_CUBIC)

        return rotated
    else:
        print("Not a single face")
        return None

argumentCount = len(sys.argv) - 1
if argumentCount != 2:
    print("usage: python alignFace.py [destinationFolderPath] [sourceFolderPath]")
else:
    destinationFolderPath = sys.argv[1]
    folderPath = sys.argv[2]
    os.mkdir(destinationFolderPath)
    filenum = 0
    for file in os.listdir(folderPath):
        if file.endswith(".jpg"):
            print(os.path.join(folderPath, file))
            img = cv2.imread(os.path.join(folderPath, file))

            alignedFace = alignFace(img)

            if alignedFace is not None:
                filename = 'face' + str(filenum) + '.jpg'
                try:
                    cv2.imwrite(os.path.join(destinationFolderPath, filename), alignedFace)
                except:
                    print("An exception occurred with " + filename)
                print(filename)
                filenum += 1