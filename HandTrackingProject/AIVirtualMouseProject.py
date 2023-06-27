import cv2
import mediapipe as mp
import time
import numpy as np
import HandTrackingModule as htm
import autopy

w_cam, h_cam = 640, 480
ptime = 0
cap = cv2.VideoCapture(0)
cap.set(3,w_cam)
cap.set(4,h_cam)
detector = htm.HandDetector()
while True:
    #find hand landmarks
    success, img = cap.read()
    img = detector.find_hands(img)
    lm_list = detector.find_position(img)





    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    cv2.putText(img,f'fps: {str(int(fps))}',(20,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)



    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

