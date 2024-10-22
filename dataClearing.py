import cv2 as cv
import mediapipe as mp
import os
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode = True, min_detection_confidence = 0.3)

DATA_DIRECTORY = './Train/Q'

data = []
labels = []

for imagePath in os.listdir(DATA_DIRECTORY)[:200]:
    img = cv.imread(os.path.join(DATA_DIRECTORY, imagePath))
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLandmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                imgRGB,
                handLandmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        plt.figure()
        plt.imshow(imgRGB)

plt.show()


