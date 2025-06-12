import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Load model dan haar cascade
model = load_model("CNN1_Sc2_converted (1).keras")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
labels = ["Tidak Mengantuk", "Mengantuk"]

# Ukuran input model
input_size = (model.input_shape[1], model.input_shape[2])


# Fungsi deteksi wajah dan prediksi kantuk
def detect_drowsiness(frame):
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        # Konversi ke grayscale
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, input_size)
        face_normalized = face_resized / 255.0
        face_reshaped = np.expand_dims(face_normalized, axis=(0, -1))  # (1, 64, 64, 1)

        prediction = model.predict(face_reshaped)[0]
        threshold = 0.50
        prob_mengantuk = prediction[0]

        if prob_mengantuk <= threshold:
            label = f"Mengantuk ({prob_mengantuk*100:.2f}%)"
            color = (0, 0, 255)
        else:
            label = f"Tidak Mengantuk ({(1 - prob_mengantuk)*100:.2f}%)"
            color = (0, 255, 0)


        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return frame


# Streamlit UI
st.title("🧠 Deteksi Kantuk Real-time")
run = st.checkbox('Jalankan Kamera')

FRAME_WINDOW = st.image([])

cap = None
if run:
    cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.warning("Gagal menangkap video.")
        break

    frame = cv2.flip(frame, 1)  # Flip horizontal
    processed_frame = detect_drowsiness(frame)
    FRAME_WINDOW.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))

if cap:
    cap.release()
