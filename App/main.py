import streamlit as st
import librosa
import numpy as np
import noisereduce as nr
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from io import BytesIO
import soundfile as sf

# Importer les fonctions de votre fichier processing.py
from audio_processing import processing, N_MFCC

# Charger le modèle TensorFlow
model = tf.keras.models.load_model('Projet_3721.keras')

def main():
    st.markdown("<h1 style='text-align: center;'>Bienvenue à notre Application de Reconnaissance Vocale des chiffres de 0 à 9</h1>", unsafe_allow_html=True)
    st.write("Enregistrez un audio de vous prononçant un chiffre compris entre 0 et 9 et nous vous dirons de quel chiffre il s'agit.")

    # Enregistrement audio via Streamlit
    audio_file = st.file_uploader("Téléchargez un fichier audio", type=["wav"])

    if audio_file is not None:
        # Lire le fichier audio téléchargé
        audio_data, sample_rate = sf.read(audio_file, dtype="float32")

        # Prétraitement de l'audio
        # st.write(sample_rate)
        # st.write(audio_data.dtype)
        X = processing(audio_data, sample_rate)

        #Vérifier et ajuster la forme des données
        if X.ndim == 1:
            X = np.expand_dims(X, axis=0)
        if X.shape[1] != N_MFCC:
            st.error(f"Les données prétraitées doivent avoir {N_MFCC} caractéristiques, mais ont {X.shape[0]}")
            return

        # Prédiction avec le modèle
        st.write(f"Shape :{X.shape}\n\n")
        pred = model.predict(X)
        pred = np.argmax(pred, axis = -1)

        # Afficher la prédiction
        st.write(f"Le chiffre prononcé est : {pred[0]}")

if __name__ == "__main__":
    main()
