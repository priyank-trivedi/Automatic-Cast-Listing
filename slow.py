from imutils.video import FPS
import numpy as np
import argparse
import imutils
import cv2
#import skvideo.io


ap = argparse.ArgumentParser()
ap.add_argument("-v","--video", required=True, help="path of video file")
args = vars(ap.parse_args())

print(args["video"])
stream = cv2.VideoCapture(args["video"])

if stream.isOpened() == False:
    print("Error")

