from flask import Flask, request, render_template, jsonify
import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * \
    1024 * 1024  # Allow up to 10 MB uploads

UPLOAD_FOLDER = 'uploads'
RECORD_FOLDER = 'recordings'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RECORD_FOLDER, exist_ok=True)

# Load model and setup
model = load_model("emotion_model.keras")
max_pad_len = 174
emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry',
                  'fearful', 'disgust', 'surprised', 'romantic', 'nervous']
le = LabelEncoder()
le.fit(emotion_labels)


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


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('audio')
        if file and (file.filename.endswith('.wav') or file.filename.endswith('.mp3')):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            features = extract_features(filepath)
            features = features[np.newaxis, ..., np.newaxis]
            prediction = model.predict(features)
            predicted_label = le.inverse_transform([np.argmax(prediction)])[0]
            return render_template('result.html', emotion=predicted_label)
    return render_template('index.html')


@app.route('/record', methods=['POST'])
def record():
    print("FILES:", request.files)
    print("FORM:", request.form)
    file = request.files.get('audio')
    if file and (file.filename.endswith('.wav') or file.filename.endswith('.mp3')):
        filename = secure_filename(file.filename)
        filepath = os.path.join(RECORD_FOLDER, filename)
        file.save(filepath)
        features = extract_features(filepath)
        features = features[np.newaxis, ..., np.newaxis]
        prediction = model.predict(features)
        predicted_label = le.inverse_transform([np.argmax(prediction)])[0]
        return jsonify({'emotion': predicted_label})
    return jsonify({'error': 'No valid audio file received'}), 400


if __name__ == '__main__':
    app.run(debug=True, threaded=True)
