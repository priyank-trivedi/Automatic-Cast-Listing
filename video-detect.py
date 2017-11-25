from imutils.video import FileVideoStream
from imutils import face_utils
import sys
import datetime
import argparse
import math, operator
import imutils
import time
import dlib
import cv2
import numpy as np
from threading import Thread
from queue import Queue
from main import get_dict
import pandas as pd
import datetime


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
ap.add_argument("-v", "--video", required=True)
ap.add_argument("-c", "--csv", required=True)
args = vars(ap.parse_args())

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

fvs = FileVideoStream(args["video"]).start()
time.sleep(1.0)
frame_count = 0
df_pos = 0


def getTimeFromStr(timeStr):
    try:
        return datetime.datetime.strptime(timeStr, "%H:%M:%S.%f").time()
    except:
        return datetime.datetime.strptime(timeStr, "%H:%M:%S").time()

def rmsdiff(im1, im2):
    img = im1-im2
    h,bins = np.histogram(img.ravel(),256,[0,256])
    #h2,bins = np.histogram(im2.ravel(),256,[0,256])
    #h = h1-h2
    sq = (value*(idx**2) for idx, value in enumerate(h))
    sum_of_squares = sum(sq)
    rms = math.sqrt(sum_of_squares/float(im1.shape[0] * im1.shape[1]))
    return rms

csv_df = pd.read_csv(args["csv"], encoding='utf-8')
csv_df = csv_df.drop_duplicates(subset=['1'])
df_length = len(csv_df)
stimes = list(map(lambda x: getTimeFromStr(x), csv_df['1'].as_matrix()))
etimes = list(map(lambda x: getTimeFromStr(x), csv_df['2'].as_matrix()))
names = csv_df['3'].as_matrix()
#print(csv_df)

prev_diff_array = None
flag =0 
fp = open('temp.txt','a')

final_dict = get_dict()

print("Total frame count:", fvs.stream.get(cv2.CAP_PROP_FRAME_COUNT))
fps = round(fvs.stream.get(cv2.CAP_PROP_FPS))
print("FPS:", fps)

def getSpeakerName(time):
    timeObj1 = time
    timeObj2 = time
    for iter in range(len(stimes)):
        if stimes[iter] < time.time() and etimes[iter] > time.time():
            return names[iter]
    return "speaker_unknown!"

while fvs.more():
    frame = fvs.read()
    #print("Frame time:", fvs.stream.get(cv2.CAP_PROP_POS_MSEC))
    #curr_time = time.strftime("%H:%M:%S", time.gmtime(frame_count/fps))
    curr_time = datetime.datetime.utcfromtimestamp(frame_count/fps)
    print(curr_time)
    #print("Frame number:", frame_count, "\nTime:", curr_time)
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #frame = np.dstack([frame, frame, frame])

    ## detection here
    rects = detector(gray, 1)

    diff_array = 0
    for (i, rect) in enumerate(rects):
        
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        #cv2.putText(frame, "Face#{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)

        #for (x, y) in shape:
        #    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        curr_data = None
        clone = frame.copy()
        cv2.putText(clone, "lips", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
        for (x,y) in shape[48:68]:
            cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
            (x, y, w, h) = cv2.boundingRect(np.array([shape[48:68]]))
            roi = frame[y:y + h, x:x + w]
            roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
            
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        res = cv2.resize(roi_gray,(100,250),interpolation = cv2.INTER_CUBIC)
        curr_data = res
        if flag==0:
            flag = 1
            prev_data = curr_data
            prev_diff_array = diff_array

        
        if flag==1:
            #if len(prev_data)==len(curr_data):
                diff_array = rmsdiff(prev_data,curr_data)
                #print(diff_array)
                thres = abs(prev_diff_array - diff_array)
                fp.write(str(prev_diff_array))
                fp.write(str(diff_array))
                #print(thres)
                if thres>=50:
                    cv2.putText(frame, "Non-Speaking", (x - 10, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    speakerName = getSpeakerName(curr_time)
                    #df_pos += ifNext
                    cv2.putText(frame, speakerName, (x - 10, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)



        #cv2.imshow("ROI", roi)
        prev_data = curr_data
        prev_diff_array = diff_array
    #output = face_utils.visualize_facial_landmarks(frame, shape)
    frame_count += 1
    cv2.imshow("Frame ", frame)
    cv2.waitKey(1)

cv2.destroyAllWindows()
fvs.stop()

