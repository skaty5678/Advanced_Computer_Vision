import cv2
import numpy as np
import time
import PoseEstimationModule as pm

cap = cv2.VideoCapture('AITrainer/1.mp4')
detector = pm.PoseDetector()
count = 0
direction = 0
ptime = 0
while True:
    success, img = cap.read()
    # img = cv2.imread('AITrainer/dips2.jpg')
    img = detector.find_pose(img,draw=False)
    lm_list = detector.find_position(img,draw=False)
    # print(lm_list)
    if len(lm_list) != 0:
        # #right arm
        # detector.find_angle(img,12,14,16)

        # left arm
        angle = detector.find_angle(img,11,13,15)
        percentage = np.interp(angle, (210,310),(0,100))
        # print(angle, percentage)

        # check for the dumbbell curls
        color = (255, 0, 255)
        if percentage == 100:
            color = (0, 255, 0)
            if direction == 0:
                count += 0.5
                direction = 1
        if percentage == 0:
            color = (0, 255, 0)
            if direction == 1:
                count += 0.5
                direction = 0
        print(count)

        cv2.rectangle(img,(0,220),(140,360),color,cv2.FILLED)
        cv2.putText(img,f"{int(count)}",(45,315),cv2.FONT_HERSHEY_PLAIN,5,(255,0,0),7)

    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    cv2.putText(img,f"FPS: {str(int(fps))}",(10,40),cv2.FONT_HERSHEY_PLAIN,2,(0,255,255),3)
    cv2.imshow('img',img)




    if cv2.waitKey(1) & 0xFF == ord('q'):
        break