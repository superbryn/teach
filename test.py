import pickle
import cv2 as cv
import mediapipe as mp
import numpy as np
import os
import customtkinter as ctk
from tkinter import filedialog, messagebox

#model 
with open('model.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

def process_image(image_path): #this basically the same thing from the webcam model
                               #but use tkinter to upload file instead of live
    img = cv.imread(image_path)
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    results = hands.process(imgRGB)
    dataAUX = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_ = []
            y_ = []

            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            for landmark in hand_landmarks.landmark:
                dataAUX.append(landmark.x - min(x_))
                dataAUX.append(landmark.y - min(y_))
            break
    
    return np.array(dataAUX) if dataAUX else None

def predict_image():
    image_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if not image_path:
        return  #closing the file selection dialog box
    features = process_image(image_path)
    if features is not None:
        prediction = model.predict([features])
        messagebox.showinfo("Prediction", f"Predicted label: {prediction[0]}")
    else:
        messagebox.showwarning("Warning", "No hand detected in the image.")

# main custontkinter configuration
ctk.set_appearance_mode('dark')
app = ctk.CTk()
app.title("Hand Sign Prediction")
app.iconbitmap(os.path.abspath('logo.ico'))
app.geometry("720x480")

upload_button = ctk.CTkButton(app, text="Upload Hand Image", command=predict_image)
upload_button.pack(pady=20)

app.mainloop()
