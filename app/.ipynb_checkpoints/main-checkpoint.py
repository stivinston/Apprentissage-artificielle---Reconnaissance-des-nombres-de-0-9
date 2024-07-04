from fastapi import FastAPI, HTTPException, File, UploadFile
#from joblib import load
import tensorflow as tf
from audio_processing import processing, N_MFCC
from pathlib import Path
import numpy as np
import io

app = FastAPI()
#model=load("model_47.33_2024-06-19 21:19:51.814168.joblib")
model=tf.keras.models.load_model("Projet_3721.keras")

def is_filepath(filepath: str) -> bool:
    return Path(filepath).is_file()

@app.post("/model_call/")
async def model_call(file: UploadFile = File(...)):
    #Verifier le type de fichier en telechargé
    if file.content_type != "audio/wav":
        raise HTTPException(status_code=400, detail="Type de fichier non supporté \n(le fichier doit être de type '.wav')")
    
    #Recuperer l'audio telechargé
    audio = await file.read()
    audio_file = io.BytesIO(audio)

    X=processing(audio_file)

    # Vérifier et ajuster la forme des données
    if X.ndim == 1:
        X = np.expand_dims(X, axis=0)
    if X.shape[1] != N_MFCC:
        raise HTTPException(status_code=400, detail=f"Les données prétraitées doivent avoir 40 caractéristiques, mais ont {X.shape[1]}")
    pred=model.predict(X)
    pred=np.argmax(pred)

    return {"result":f"{pred}"}