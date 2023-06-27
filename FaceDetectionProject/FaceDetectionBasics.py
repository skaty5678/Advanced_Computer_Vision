import cv2
import mediapipe as mp
import time

mp_face_detection = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection()

cap = cv2.VideoCapture(0)
ptime = 0

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)
    print(results)

    if results.detections:
        for id, detection in enumerate(results.detections):
            bbox_c = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = (
                int(bbox_c.xmin * iw),
                int(bbox_c.ymin * ih),
                int(bbox_c.width * iw),
                int(bbox_c.height * ih)
            )
            cv2.rectangle(img, bbox, (255, 0, 255), 2)
            cv2.putText(
                img,
                f"{int(detection.score[0]*100)}%",
                (bbox[0], bbox[1] - 20),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (255, 0, 255),
                2
            )

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv2.putText(
        img,
        f"FPS: {int(fps)}",
        (20, 70),
        cv2.FONT_HERSHEY_PLAIN,
        3,
        (0, 255, 0),
        3
    )
    cv2.imshow('face', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
