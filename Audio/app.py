import streamlit as st
import sounddevice as sd
import librosa
import numpy as np
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


model = tf.keras.models.load_model('speech-emotion.keras')

# Define your emotion labels in the order your model predicts
emotion_labels = ['disgust', 'ps', 'happy', 'sad', 'neutral', 'fear', 'angry']  

# App title
st.title("🎙️ Real-Time Emotion Detection App")
st.markdown("Speak into your mic and let the model predict your emotion.")

# Record and predict button
if st.button("🎧 Record & Predict"):
    st.write("Recording for 3 seconds... Please speak.")
    
    duration = 3 # seconds
    sr = 22050

    # Record audio from mic
    recording = sd.rec(int(duration * sr), samplerate = sr, channels=1)
    sd.wait()

    # Convert to 1D array
    audio = recording.flatten()

    # Feature extraction (MFCC)
    try:
        mfcc = librosa.feature.mfcc(y=audio, sr = sr, n_mfcc=40)
        mfcc_scaled = np.mean(mfcc.T, axis=0).reshape(1, -1)

        # Predict emotion
        prediction = model.predict(mfcc_scaled)
        predicted_index = np.argmax(prediction)
        predicted_emotion = emotion_labels[predicted_index]

        st.success(f"🎉 Predicted Emotion: **{predicted_emotion.capitalize()}**")
    except Exception as e:
        st.error(f"Error processing audio: {e}")
