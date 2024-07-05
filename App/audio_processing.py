import librosa
import numpy as np
import noisereduce as nr
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (-1,1))

N_MFCC=40

def MinMaxScale(a):
    a = scaler.fit_transform(a)
    return a

def extract_silence_StartEnd(audio, sr):
    noise_profile = audio[:sr]  
    reduced_noise = nr.reduce_noise(y=audio, sr=sr, y_noise=noise_profile)
    audio_trim,_=librosa.effects.trim(reduced_noise, top_db=10)
    clean_audio = librosa.util.normalize(audio_trim)
    return clean_audio


def processing(audio, sr):
    clean_audio=extract_silence_StartEnd(audio, sr)
    mfcc = librosa.feature.mfcc(y=clean_audio, sr=sr, n_mfcc=40)
    X = np.mean(mfcc, axis=1)
    print("*"*10)
    print(f"\n\n{X.shape[0]}\n\n")
    print("*"*10)
    #Normalisation minmax
    #X = MinMaxScale(X)
    X_reshape = np.reshape(X, (1, -1))
    return X_reshape