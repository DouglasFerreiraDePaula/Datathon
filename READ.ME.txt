
# MatchAI – DataThon Final Project

## 🎯 Objetivo
Construir uma solução de Machine Learning para classificar candidatos com maior potencial de compatibilidade com as vagas disponíveis, utilizando grandes volumes de dados de recrutamento.

---

## 🧱 Estrutura do Projeto

```
TECH FINAL/
├── data/                 # Arquivos JSON brutos
├── models/               # Modelo treinado (.pkl)
├── src/                  # Código-fonte principal (treinamento e API)
├── test/                 # Testes automatizados
├── requirements.txt      # Dependências do projeto
├── Dockerfile            # Empacotamento com Docker
├── README.md             # Documentação para GitHub
```

---

## 🛠️ Etapas Técnicas

### 1. Carregamento de dados
- Leitura eficiente dos arquivos:
  - applicants.json
  - jobs.json
  - prospects.json
- Junção lógica e seleção de atributos com `data_loader.py`.

### 2. Pré-processamento
- Tratamento de texto (`cv_texto`, `competencias`, etc.) com `TfidfVectorizer`.
- Codificação de variáveis categóricas com `OneHotEncoder`.
- Preparação dos dados via `ColumnTransformer`.

### 3. Treinamento do modelo
- Algoritmo: `RandomForestClassifier`.
- Pipeline completo com `train_model.py`.
- Modelo salvo como `models/model.pkl`.

### 4. API com FastAPI
- Endpoint POST `/predict` implementado em `main.py`.
- Recebe JSON com perfil do candidato e retorna previsão (`0` ou `1`).

### 5. Testes
- Teste automatizado com `pytest` em `test/test_api.py`.
- Verifica integridade da rota e resposta.

### 6. Docker
- Dockerfile para build e execução da API com `uvicorn`.

---

## ▶️ Como executar

### Ambiente local (VS Code):

```bash
# Ativar ambiente virtual
venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt

# Treinar modelo
python src/train_model.py

# Iniciar API
uvicorn src.main:app --reload
```

### Testar a API manualmente:

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "nivel_academico": "Ensino Superior Completo",
    "nivel_ingles": "Avançado",
    "nivel_espanhol": "Básico",
    "area_atuacao": "TI",
    "nivel_profissional": "Pleno",
    "cv_texto": "Engenheiro de software com experiência em Python e APIs REST",
    "atividades": "Desenvolvimento backend",
    "competencias": "Python, Django, Git",
    "sap": "Sim"
  }'
```

---

## 🧪 Testes

```bash
pytest test/test_api.py
```

---

## 🐳 Docker

```bash
docker build -t matchai-api .
docker run -p 8000:8000 matchai-api
```

---

## 📌 Resultado

- API funcional e escalável
- Classificador robusto de candidatos com base em dados reais
- Pipeline automatizado para ingestão, processamento, modelagem e deploy
