import cv2 as cv
import mediapipe as mp
import numpy as np
import pickle

# Load MediaPipe and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Load trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Set up webcam
capture = cv.VideoCapture(0)

# Initialize MediaPipe Hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Labels dictionary
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
    dataAUX = []
    x_ = []
    y_ = []

    ret, frame = capture.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame to RGB
    frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Process the frame for hand landmarks
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

            # Collect raw x and y values
            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            # Normalize and add to dataAUX (same as training)
            for landmark in hand_landmarks.landmark:
                dataAUX.append(landmark.x - min(x_))
                dataAUX.append(landmark.y - min(y_))

        # Convert to numpy array and reshape if necessary
        dataAUX = np.asarray(dataAUX).reshape(1, -1)

        # Make a prediction
        prediction = model.predict(dataAUX)

        # Get the predicted character from labelDict
        predicted_character = labelDict[prediction[0]]

        # Draw bounding box and label around the hand
        H, W, _ = frame.shape
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) + 10
        y2 = int(max(y_) * H) + 10

        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv.putText(frame, predicted_character, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv.LINE_AA)

    # Display the frame
    cv.imshow('frame', frame)

    # Break on 'q' key press
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close windows
capture.release()
cv.destroyAllWindows()
