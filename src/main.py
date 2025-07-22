from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("models/model.pkl")

class CandidateInput(BaseModel):
    nivel_academico: str
    nivel_ingles: str
    nivel_espanhol: str
    area_atuacao: str
    nivel_profissional: str
    cv_texto: str
    atividades: str
    competencias: str
    sap: str

@app.post("/predict")
def predict(input_data: CandidateInput):
    df = pd.DataFrame([input_data.dict()])
    prediction = model.predict(df)[0]
    return {"prediction": int(prediction)}