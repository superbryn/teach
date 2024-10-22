import cv2 as cv
import mediapipe as mp
import os
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode = True, min_detection_confidence = 0.3)

DATA_DIRECTORY = './prototype'

data = []
labels = []

for Dir in os.listdir(DATA_DIRECTORY):
    for imagePath in os.listdir(os.path.join(DATA_DIRECTORY, Dir))[:200]:
        dataAUX = []
        img = cv.imread(os.path.join(DATA_DIRECTORY, Dir, imagePath))
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        x_ = []
        y_ = []

        results = hands.process(imgRGB)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                for landmark in hand_landmarks.landmark:
                    dataAUX.append(landmark.x - min(x_))
                    dataAUX.append(landmark.y - min(y_))

            data.append(dataAUX)
            labels.append(Dir)

'''
            for handLandmarks in results.multi_hand_landmarks:
                for i in range(len(handLandmarks.landmark)):
                    x = handLandmarks.landmark[i].x
                    y = handLandmarks.landmark[i].y
                    dataAUX.append(x)
                    dataAUX.append(y)

                     '''

with open("data.pickle", 'wb') as f:
    import pickle
    pickle.dump({'data':data, 'labels':labels},f)