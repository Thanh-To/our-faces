import subprocess
import os
import sys

new_dir_name = 'averages';

argumentCount = len(sys.argv) - 1
if argumentCount != 1:
    print("usage: python3 generateAverages.py [testDataPath]")
else:
    testDataPath = sys.argv[1]

    for subdir, dirs, files in os.walk(testDataPath):

    	if len(dirs) > 0:
    		os.mkdir(new_dir_name)

    	for dir in dirs:

    		sourceFolderPath = os.path.join(subdir, dir)

    		print("Processing " + sourceFolderPath)

    		print("Finding all faces")
    		subprocess.call("python3 findFace.py " + dir + " " + sourceFolderPath, shell=True)

    		print("Extracting all facial features")
    		subprocess.call("python3 ../average-faces-opencv/extract.py ../average-faces-opencv/shape_predictor_68_face_landmarks.dat " + dir, shell=True)

    		print("Generating average photo")
    		subprocess.call("python3 ../average-faces-opencv/average.py " + dir, shell=True)

    		print("Renaming average face file")
    		os.rename(r'average_face.jpg',r''+new_dir_name+'/average_face_'+dir+'.jpg')