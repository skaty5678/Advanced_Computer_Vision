import cv2
import mediapipe as mp
import time


class FaceDetector:
    def __init__(self, min_detection_con=0.5):
        self.min_detection_con = min_detection_con
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_draw = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(self.min_detection_con)

    def find_faces(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face_detection.process(img_rgb)
        bboxes = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bbox_c = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = (
                    int(bbox_c.xmin * iw),
                    int(bbox_c.ymin * ih),
                    int(bbox_c.width * iw),
                    int(bbox_c.height * ih)
                )
                bboxes.append([id, bbox, detection.score])
                if draw:
                    img = self.fancy_draw(img, bbox)
                    cv2.putText(
                        img,
                        f"{int(detection.score[0]*100)}%",
                        (bbox[0], bbox[1] - 20),
                        cv2.FONT_HERSHEY_PLAIN,
                        2,
                        (255, 0, 255),
                        2
                    )
        return img, bboxes

    def fancy_draw(self, img, bbox, l=30, t=5, rt=1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h
        cv2.rectangle(img, bbox, (255, 0, 255), rt)

        cv2.line(img, (x, y), (x + l, y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y + l), (255, 0, 255), t)

        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y + l), (255, 0, 255), t)

        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)

        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)

        return img


def main():
    cap = cv2.VideoCapture('face_videos/1.mp4')
    ptime = 0
    detector = FaceDetector()
    while True:
        success, img = cap.read()
        img, bboxes = detector.find_faces(img)
        print(bboxes)
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


if __name__ == '__main__':
    main()

