import cv2
import time
from HandTrackingModule import HandDetector

# Initialize variables for FPS calculation
p_time = 0
c_time = 0

# Open the video capture
cap = cv2.VideoCapture(0)

# Create an instance of the HandDetector class
detector = HandDetector()

while True:
    # Read the video frames
    success, img = cap.read()

    # Find hands in the frame
    img = detector.find_hands(img)

    # Find landmark positions on the hands
    lm_list = detector.find_position(img)

    if len(lm_list) != 0:
        print(lm_list[4])

    # Calculate and display the FPS
    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    # Display the image with hand landmarks
    cv2.imshow('Hand Landmarks', img)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
