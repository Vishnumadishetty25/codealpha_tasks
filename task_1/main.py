import os
import random
import numpy as np
import librosa
import sounddevice as sd
import scipy.io.wavfile as wav
from tkinter import Tk, Label, Button, StringVar, Toplevel
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Team codealpha this is the Configuration part
dataset_path = r"C:\Users\sruja\OneDrive\Desktop\project\ravdess"
model_path = "emotion_model.keras"
max_pad_len = 174
emotion_labels = ['neutral', 'calm', 'happy', 'sad',
                  'angry', 'fearful', 'disgust', 'surprised']
le = LabelEncoder()
le.fit(emotion_labels)

# Here we Load model
model = load_model(model_path)

# Feature Extraction is Done Here


def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    if mfccs.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=(
            (0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]
    return mfccs


def play_audio(file_path):
    audio, sr = librosa.load(file_path, sr=22050)
    print("[🔊] Playing audio...")
    sd.play(audio, samplerate=sr)
    sd.wait()

#  Random File Prediction Done Here


def predict_random_file():
    wav_files = [os.path.join(root, f)
                 for root, _, files in os.walk(dataset_path)
                 for f in files if f.endswith('.wav')]
    selected_file = random.choice(wav_files)
    features = extract_features(selected_file).reshape(1, 40, max_pad_len)
    prediction = model.predict(features)
    predicted_emotion = le.inverse_transform([np.argmax(prediction)])[0]
    file_label.set(f"File: {os.path.basename(selected_file)}")
    emotion_label.set(f"Predicted Emotion: {predicted_emotion}")
    print(f"\n[✔] Random File: {selected_file}")
    print(f"[✔] Predicted Emotion: {predicted_emotion}")
    play_audio(selected_file)

#  Live Mic Prediction GUI Set Up


def start_microphone_gui():
    mic_win = Toplevel()
    mic_win.title("Live Mic Emotion Prediction")
    mic_win.geometry("400x250")

    status = StringVar()
    result = StringVar()

    def record_and_predict():
        duration = 3
        fs = 22050
        status.set("Recording... Speak now")
        mic_win.update()
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        wav.write("live_recording.wav", fs, recording)

        features = extract_features(
            "live_recording.wav").reshape(1, 40, max_pad_len)
        prediction = model.predict(features)
        predicted_emotion = le.inverse_transform([np.argmax(prediction)])[0]
        result.set(f"Predicted Emotion: {predicted_emotion}")

        # Playback
        status.set("Playing your voice...")
        audio = recording.flatten()
        sd.play(audio, samplerate=fs)
        sd.wait()
        status.set("Done.")

    Label(mic_win, text="Live Mic Emotion Prediction",
          font=("Arial", 14)).pack(pady=10)
    Label(mic_win, textvariable=status, font=("Arial", 12)).pack(pady=5)
    Label(mic_win, textvariable=result, font=(
        "Arial", 12, "bold")).pack(pady=5)
    Button(mic_win, text="Record & Predict",
           command=record_and_predict, font=("Arial", 12)).pack(pady=15)

#  Main GUI


def start_gui():
    app = Tk()
    app.title("Speech Emotion Recognizer")
    app.geometry("400x300")

    global file_label, emotion_label
    file_label = StringVar()
    emotion_label = StringVar()

    Label(app, text="Speech Emotion Detection",
          font=("Arial", 16)).pack(pady=10)
    Label(app, textvariable=file_label, font=("Arial", 12)).pack(pady=5)
    Label(app, textvariable=emotion_label, font=(
        "Arial", 12, "bold")).pack(pady=5)

    Button(app, text="Predict from Random File",
           command=predict_random_file, font=("Arial", 12)).pack(pady=10)
    Button(app, text="Live Mic Emotion GUI",
           command=start_microphone_gui, font=("Arial", 12)).pack(pady=10)

    app.mainloop()


# Terminal Prompt
if __name__ == "__main__":
    print("========== Emotion Recognition ==========")
    print("1. Launch GUI (includes Live Mic + Random Audio)")
    choice = input("Choose (1): ").strip()

    if choice == "1":
        start_gui()
    else:
        print("Invalid choice. Please run again.")
