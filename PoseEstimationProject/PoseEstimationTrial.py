import cv2
import time
import PoseEstimationModule as pm


cap = cv2.VideoCapture('pose_videos/2.mp4')
ptime = 0
detector = pm.PoseDetector()


while True:
    success, img = cap.read()
    img = detector.find_pose(img)
    lmlist = detector.find_position(img, draw=False)
    if len(lmlist) != 0:
        print(lmlist[14])
        cv2.circle(img, (lmlist[14][1], lmlist[14][2]), 15, (0, 0, 255), cv2.FILLED)

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow('video', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
