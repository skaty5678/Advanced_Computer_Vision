import cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
draw_specs = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture('face_videos/2.mp4')
ptime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, face_landmarks, mpFaceMesh.FACEMESH_CONTOURS, landmark_drawing_spec=draw_specs)
            for id, lm in enumerate(face_landmarks.landmark):
                ih, iw, ic = img.shape
                x, y = int(lm.x * iw), int(lm.y * ih)
                print(id, x, y)

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv2.putText(img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    cv2.imshow('video', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
