import collections

import mediapipe as mp
import numpy as np
import cv2
from loguru import logger

# processing pipeline auf vid schnipsel ausführen
# modell predict
# ausgabe stream, skelleton, prediction

sliding_window = 120
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

def run():
    counter = 0
    window1 = collections.deque(maxlen=80)
    window2 = collections.deque(maxlen=120)
    cv2.namedWindow("Gesture Recognition")
    # Stream öffnen
    cam = cv2.VideoCapture(0)
    ret_val, img = cam.read()
    width, height, d = img.shape
    skeleton = np.zeros([width, height, 3], dtype="uint8")
    skeletons = np.zeros([width, height, 3], dtype="uint8")
    M = np.ones([width, height, 3], dtype="uint8")*20
    while True:
        ret_val, img = cam.read()
        # sliding window definieren
        window1.append(img)
        window2.append(img)
        counter += 1
        if counter == 10:
            counter = 0
            # predict windows
            # get skelleton for active image
            skeleton = get_skeleton(img)
            skeletons = cv2.subtract(skeletons, M)
            skeletons = cv2.add(skeletons, skeleton)
        # concat images
        preview = np.concatenate((img, skeleton), axis=1)
        preview = np.concatenate((preview, skeletons), axis=1)
        preview = cv2.resize(preview, (0, 0), fx=0.7, fy=0.7)
        cv2.imshow('Gesture Recognition', preview)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    pass


def get_skeleton(image):
    width, height, d = image.shape
    mask = np.zeros([width, height, 3], dtype="uint8")
    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        with mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as pose:

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        mask,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
            # Flip the image horizontally for a selfie-view display.
            results = pose.process(image)
            mp_drawing.draw_landmarks(
                mask,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    return cv2.flip(mask, 1)


def stop():
    cv2.DestroyAllWindows()
    pass


if __name__ == "__main__":
    run()
