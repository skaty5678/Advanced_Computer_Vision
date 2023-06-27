import cv2
import mediapipe as mp
import time


class FaceMeshDetector:
    def __init__(self, static_mode=False, max_faces=2, min_detection_con=0.5, min_track_con=0.5):
        self.static_mode = static_mode
        self.max_faces = max_faces
        self.min_detection_con = min_detection_con
        self.min_track_con = min_track_con

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.static_mode, self.max_faces)
        self.draw_specs = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def find_face_mesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for face_landmarks in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, face_landmarks, self.mpFaceMesh.FACEMESH_CONTOURS,
                                               landmark_drawing_spec=self.draw_specs)
                face = []
                for id, lm in enumerate(face_landmarks.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 255, 0), 1)
                    face.append((x, y))
                faces.append(face)
        return img, faces


def main():
    cap = cv2.VideoCapture(0)
    ptime = 0

    detector = FaceMeshDetector()
    while True:
        success, img = cap.read()
        img, faces = detector.find_face_mesh(img)
        if len(faces) != 0:
            print(len(faces))
            ctime = time.time()
            fps = 1 / (ctime - ptime)
            ptime = ctime
            cv2.putText(img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
            cv2.imshow('video', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
