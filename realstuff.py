import cv2 as cv
import mediapipe as mp
import numpy as np
import pickle

# mediapipe 
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# load the file
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# opens webcam
capture = cv.VideoCapture(0)

# media pipe hand initialization
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# labels.. like handpipe says A its gonna say A
labelDict = {
    'A':'A',
    'B':'B',
    'C':'C',
    'D':'D',
    'E':'E',
    'F':'F',
    'G':'G',
    'H':'H',
    'I':'I',
    'J':'J',
    'K':'K',
    'L':'L',
    'M':'M',
    'N':'N',
    'O':'O',
    'P':'P',
    'Q':'Q',
    'R':'R',
    'S':'S',
    'T':'T',
    'U':'U',
    'V':'V',
    'W':'W',
    'X':'X',
    'Y':'Y',
    'Z':'Z'
}

while True:
    dataAUX = [] #aux list
    x_ = []
    y_ = []

    ret, frame = capture.read()
    if not ret:
        print("Failed to grab frame")
        break

    # coverts the frame from bgr to rgb so that mediapipe can initialize
    frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # processed handland mark
    results = hands.process(frameRGB)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # collect raw x and y values
            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            # normalize and add to dataAUX (literally or idk really same as training)
            for landmark in hand_landmarks.landmark:
                dataAUX.append(landmark.x - min(x_))
                dataAUX.append(landmark.y - min(y_))

        # convert to numpy array and reshape if necessary. reshape is done so that the x42 and x84 can coexist. you know the deal
        dataAUX = np.asarray(dataAUX).reshape(1, -1)

        # the prediction... the main head
        prediction = model.predict(dataAUX)

        # checks the predicted to the labels
        predicted_character = labelDict[prediction[0]]

        # the box around the hand
        H, W, _ = frame.shape
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) + 10
        y2 = int(max(y_) * H) + 10

        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv.putText(frame, predicted_character, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv.LINE_AA)

    # displays the frame
    cv.imshow('frame', frame)

    # if i wanna quit i can just press q
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# cv2 stuff lol
capture.release()
cv.destroyAllWindows()
