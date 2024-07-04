import librosa
import numpy as np
import noisereduce as nr

MIN=np.array([-516.85785 ,  -94.969505, -109.31393 ,  -75.63553 ,  -85.504555,
        -68.11859 ,  -59.731438,  -63.875603,  -52.936108,  -38.370995,
        -50.184464,  -37.012592,  -36.15209 ,  -44.41911 ,  -39.29241 ,
        -32.136703,  -34.305305,  -33.747932,  -29.43626 ,  -26.89443 ,
        -25.636246,  -27.290241,  -28.936737,  -19.351246,  -21.054691,
        -21.030304,  -21.970343,  -20.623749,  -23.53243 ,  -17.831923,
        -19.498661,  -19.327198,  -27.996326,  -19.90473 ,  -16.901575,
        -15.305685,  -18.733402,  -17.96081 ,  -16.253376,  -13.192233])
MAX=np.array([-67.27921 , 239.93767 ,  98.69251 , 118.30383 ,  82.380005,
        72.85535 ,  40.09782 ,  36.87512 ,  25.931875,  34.940918,
        24.160915,  22.334545,  39.09787 ,  19.995073,  25.522259,
        26.401236,  33.161068,  25.035618,  16.116268,  15.084256,
        13.379552,  19.768723,  16.832075,  26.910765,  35.250362,
        30.959858,  29.13461 ,  34.15347 ,  44.65522 ,  35.16877 ,
        28.20391 ,  32.587395,  32.76949 ,  35.963913,  32.72269 ,
        29.709745,  29.911394,  28.899542,  25.312922,  26.45787])
N_MFCC=40

def MinMaxScale(X, Min, Max):
    X=np.array(X)
    return (X-Min)/(Max-Min)

def extract_silence_StartEnd(audio, sr):
    noise_profile = audio[:sr]  
    reduced_noise = nr.reduce_noise(y=audio, sr=sr, y_noise=noise_profile)
    audio_trim,_=librosa.effects.trim(reduced_noise, top_db=20)
    clean_audio = librosa.util.normalize(audio_trim)
    return clean_audio

def processing(filepath):
    audio, sample_rate=librosa.load(filepath)
    clean_audio=extract_silence_StartEnd(audio, sample_rate)
    mfcc = librosa.feature.mfcc(y=clean_audio, sr=sample_rate, n_mfcc=N_MFCC)
    X=np.mean(mfcc, axis=1)
    print("*"*10)
    print(f"\n\n{X.shape}\n\n")
    print("*"*10)
    #Normalisation minmax
    X = MinMaxScale(X, MIN, MAX)
    return X