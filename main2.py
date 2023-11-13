import cv2
import numpy
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
mp_holistic= mp.solutions.holistic
mp_drawing=mp.solutions.drawing_utils
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image , cv2.COLOR_RGB2BGR)
    return image, results
def draw_landmarks(image, results):
    image_copy=image.copy()
    mp_drawing.draw_landmarks(image_copy, results.face_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image_copy, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image_copy, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image_copy, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    return image_copy
cap= cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():
        ret, frame= cap.read()
        image , results = mediapipe_detection(frame, holistic)
        print(results)
        cv2.imshow('OpenCV Feed', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    draw_landmarks(frame, holistic)
    plt.imshow(cv2.cvtColor(draw_landmarks(frame,results), cv2.COLOR_BGR2RGB))
