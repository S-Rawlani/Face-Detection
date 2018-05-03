import numpy as np
import cv2
import os
from os import listdir
from os.path import isfile, join
import sys


def detectShush(frame, location, ROI, cascade,nose_cascade):

    noses = nose_cascade.detectMultiScale(ROI, 1.15, 7, 0, (20, 20))
    minN=sys.maxsize
    nx = ny = nw = nh = 0
    for (mx, my, mw, mh) in noses:
        mx += location[0]
        my += location[1]
        if(minN>my + mh):
            minN=my + mh
            nx=mx
            ny=my
            nw=mw
            nh=mh
    #cv2.rectangle(frame, (nx, ny), (nx + nw, ny + nh), (255, 0, 255), 2)

    #nx += location[0]
    #ny += location[1]
    #print("nose ", nx, " ", ny, " ", nx +nw, " ", ny + nh)
    if(nx==nx +nw and ny==ny + nh):
        nx = ny = nw = nh = 0
    mouths = cascade.detectMultiScale(ROI, 1.15, 10, 0, (10, 10))
    for (mx, my, mw, mh) in mouths:
        mx += location[0]
        my += location[1]
        #print("in ",ny+nh," ",my)
        if(ny+nh<=my):

            #print("mouth ",mx," ",my," ",mx + mw," ", my + mh)
            cv2.rectangle(frame, (mx, my), (mx + mw, my + mh), (0, 0, 255), 2)
    #print(len(mouths) == 0)
    return len(mouths) == 0


def detect(frame, faceCascade, mouthsCascade,nose_cascade):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #    gray_frame = cv2.equalizeHist(gray_frame)
    #    gray_frame = cv2.medianBlur(gray_frame, 5)

    faces = faceCascade.detectMultiScale(
        gray_frame, 1.15, 4, 0 | cv2.CASCADE_SCALE_IMAGE, (40, 40))
    detected = 0
    for (x, y, w, h) in faces:
        # ROI for mouth
        x1 = x
        h2 = int(h / 2)
        y1 = y + h2
        mouthROI = gray_frame[y1:y1 + h2, x1:x1 + w]

        if detectShush(frame, (x1, y1), mouthROI, mouthsCascade,nose_cascade):
            detected += 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #print("--------in----------")
    return detected


def run_on_folder(cascade1, cascade2,cascade3, folder):
    if (folder[-1] != "/"):
        folder = folder + "/"
    files = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]
    windowName = None
    totalCnt = 0
    for f in files:
        img = cv2.imread(f)
        if type(img) is np.ndarray:
            lCnt = detect(img, cascade1, cascade2,cascade3)
            totalCnt += lCnt
            if windowName != None:
                cv2.destroyWindow(windowName)
            windowName = f
            cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(windowName, img)
            cv2.waitKey(0)
    return totalCnt


def runonVideo(face_cascade, eyes_cascade,nose_cascade):
    videocapture = cv2.VideoCapture(0)
    if not videocapture.isOpened():
        print("Can't open default video camera!")
        exit()

    windowName = "Live Video"
    showframe = True
    while (showframe):
        ret, frame = videocapture.read()

        if not ret:
            print("Can't capture frame")
            break
        detect(frame, face_cascade, eyes_cascade,nose_cascade)
        cv2.imshow(windowName, frame)
        if cv2.waitKey(30) >= 0:
            showframe = False

    videocapture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # check command line arguments: nothing or a folderpath
    if len(sys.argv) != 1 and len(sys.argv) != 2:
        print(sys.argv[0] + ": got " + len(sys.argv) - 1 +
              "arguments. Expecting 0 or 1:[image-folder]")
        exit()

    # load pretrained cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    mouth_cascade = cv2.CascadeClassifier('Mouth.xml')
    nose_cascade=cv2.CascadeClassifier('Nariz.xml')
    if (len(sys.argv) == 2):  # one argument
        folderName = sys.argv[1]
        detections = run_on_folder(face_cascade, mouth_cascade,nose_cascade, folderName)
        print("Total of ", detections, "detections")
    else:  # no arguments
        runonVideo(face_cascade, mouth_cascade,nose_cascade)
