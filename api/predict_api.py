from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="MatchAI Recruiter")
model = joblib.load("modelo_match_decision.joblib")

class PredictionRequest(BaseModel):
    cv_texto: str
    atividades: str
    competencias: str
    nivel_academico: str
    nivel_ingles: str
    nivel_espanhol: str
    nivel_profissional: str
    sap: str
    area_atuacao: str

@app.post("/predict")
def predict(data: PredictionRequest):
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df).max()
    return {
        "classificacao": prediction,
        "confianca": round(proba, 4)
    }
