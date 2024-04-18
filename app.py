from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

app = Flask(__name__)

# Load scaler and label encoder
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Load classifiers
classifiers = {}
models_path = "models"  # Path to the directory containing trained models
for name in os.listdir(models_path):
    if name.endswith(".pkl"):
        model_name = name.split(".")[0]
        classifiers[model_name] = joblib.load(os.path.join(models_path, name))

# Function to extract features from audio files
def extract_features(file_path, mfcc=True, chroma=True, mel=True):
    audio, sr = librosa.load(file_path)
    features = []
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20).T, axis=0)
        features.extend(mfccs)
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T,axis=0)
        features.extend(chroma)
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T,axis=0)
        features.extend(mel)
    return features

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Extract features from uploaded audio file
        features = extract_features(file_path)

        # Standardize features
        features_scaled = scaler.transform([features])

        # Classify using all models
        results = {}
        for name, clf in classifiers.items():
            y_pred = clf.predict(features_scaled)
            results[name] = label_encoder.inverse_transform(y_pred)[0]

        return jsonify(results)

if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = 'uploads'
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    app.run(debug=True)

