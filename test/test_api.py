from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_predict_valid_input():
    payload = {
        "nivel_academico": "Ensino Superior Completo",
        "nivel_ingles": "Avançado",
        "nivel_espanhol": "Básico",
        "area_atuacao": "TI",
        "nivel_profissional": "Pleno",
        "cv_texto": "Experiência em Python e APIs REST",
        "atividades": "Desenvolvimento backend",
        "competencias": "Python, Django, REST",
        "sap": "Sim"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()