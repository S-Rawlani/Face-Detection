import numpy as np
import cv2
import os
from os import listdir
from os.path import isfile, join
import sys


def detectWink(frame, location, ROI, cascade1,cascade2,cascade3,cascade4):
    eyes = cascade1.detectMultiScale(
        ROI, 1.15, 3, 0 | cv2.CASCADE_SCALE_IMAGE, (10, 20))

    leftEyes=[]
    for e in eyes:

        xb=e[0]
        yb=e[1]
        e[0] += location[0]
        e[1] += location[1]
        x, y, w, h = e[0], e[1], e[2], e[3]
        eyeROI=ROI[yb:yb + h,xb:xb + w ]


        leftEyes = cascade2.detectMultiScale(
            eyeROI, 1.15, 3, 0 | cv2.CASCADE_SCALE_IMAGE, (10, 20))
        if(len(leftEyes)==0):

            leftEyes = cascade3.detectMultiScale(
                eyeROI, 1.15, 3, 0 | cv2.CASCADE_SCALE_IMAGE, (10, 20))
            rightEyes=cascade4.detectMultiScale(
                eyeROI, 1.15, 3, 0 | cv2.CASCADE_SCALE_IMAGE, (10, 20))
            if(len(leftEyes)>=2):

                leftEyes=[]
            elif(len(leftEyes)==1 and len(rightEyes)==1):
                el=leftEyes[0]
                er=rightEyes[0]

                if(abs(el[0]-er[0])>=10):
                    leftEyes = []

        for e1 in leftEyes:


            e1[0] += x
            e1[1] += y
            xR, yR, w, h = e1[0], e1[1], e1[2], e1[3]

            cv2.rectangle(frame, (xR, yR), (xR + w, yR + h), (0, 0, 255), 2)


    return (len(leftEyes)==1)# number of eyes is one

def detect(frame, faceCascade,bothEyeCascade,eyesCascade, eyesLeftCascade,eyesRightCascade):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = np.array(gray_frame, dtype='uint8')
    # possible frame pre-processing:
    # gray_frame = cv2.equalizeHist(gray_frame)
    # gray_frame = cv2.medianBlur(gray_frame, 5)

    scaleFactor = 1.15  # range is from 1 to ..
    minNeighbors = 4  # range is from 0 to ..
    flag = 0 | cv2.CASCADE_SCALE_IMAGE  # either 0 or 0|cv2.CASCADE_SCALE_IMAGE
    minSize = (30, 30)  # range is from (0,0) to ..
    faces = faceCascade.detectMultiScale(
        gray_frame,
        scaleFactor,
        minNeighbors,
        flag,
        minSize)

    detected = 0
    for f in faces:
        x, y, w, h = f[0], f[1], f[2], f[3]
        faceROI = gray_frame[y:y + h, x:x + w]
        if detectWink(frame, (x, y), faceROI,bothEyeCascade,eyesCascade, eyesLeftCascade,eyesRightCascade):
            detected += 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return detected


def run_on_folder(cascade1, cascade2,cascade3,cascade4,cascade5, folder):
    if (folder[-1] != "/"):
        folder = folder + "/"
    files = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]

    windowName = None
    totalCount = 0
    for f in files:
        img = cv2.imread(f, 1)
        if type(img) is np.ndarray:
            lCnt = detect(img, cascade1, cascade2,cascade3,cascade4,cascade5)
            totalCount += lCnt

            if windowName != None:
                cv2.destroyWindow(windowName)
            windowName = f
            cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(windowName, img)
            cv2.waitKey(0)
    return totalCount


def runonVideo(face_cascade,bothEye_cascade,eye_cascade, eyesLeft_cascade,eyesRight_cascade):
    videocapture = cv2.VideoCapture(0)
    if not videocapture.isOpened():
        print("Can't open default video camera!")
        exit()

    windowName = "Live Video"
    showlive = True
    while (showlive):
        ret, frame = videocapture.read()

        if not ret:
            print("Can't capture frame")
            exit()

        detect(frame, face_cascade, bothEye_cascade,eye_cascade,eyesLeft_cascade,eyesRight_cascade)
        cv2.imshow(windowName, frame)
        if cv2.waitKey(30) >= 0:
            showlive = False

    # outside the while loop
    videocapture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # check command line arguments: nothing or a folderpath
    if len(sys.argv) != 1 and len(sys.argv) != 2:
        print(sys.argv[0] + ": got " + len(sys.argv) - 1
              + "arguments. Expecting 0 or 1:[image-folder]")
        exit()

    # load pretrained cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades
                                         + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade eye.xml')
    eyeLeft_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
    eyeRight_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')

    bothEye_cascade = cv2.CascadeClassifier('frontalEyes35x16.xml')
    #eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades
    #                                    + 'haarcascade_eye.xml')
    #print(eyeLeft_cascade," ",eyeLeft_cascade)
    if (len(sys.argv) == 2):  # one argument
        folderName = sys.argv[1]
        detections = run_on_folder(face_cascade,bothEye_cascade, eye_cascade,eyeLeft_cascade,eyeRight_cascade, folderName)
        print("Total of ", detections, "detections")
    else:  # no arguments
        runonVideo(face_cascade,bothEye_cascade, eye_cascade,eyeLeft_cascade,eyeRight_cascade)