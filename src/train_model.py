import joblib
from src.data_loader import load_data
from src.preprocess import build_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

applicants_path = "data/applicants.json"
jobs_path = "data/jobs.json"
prospects_path = "data/prospects.json"

df = load_data(applicants_path, jobs_path, prospects_path)

features = ["cv_texto", "atividades", "competencias", "nivel_academico", "nivel_ingles",
            "nivel_espanhol", "nivel_profissional", "sap", "area_atuacao"]
X = df[features]
y = df["status"]

# Verificar e remover classes com menos de 2 amostras
from collections import Counter

# Conta quantas vezes cada classe aparece
y_counts = Counter(y)

# Filtra somente classes com 2 ou mais amostras
valid_classes = [label for label, count in y_counts.items() if count >= 2]

# Aplica o filtro
X = X[y.isin(valid_classes)]
y = y[y.isin(valid_classes)]

# Garante que os campos textuais são strings válidas
X["cv_texto"] = X["cv_texto"].fillna("").astype(str)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)

X = df.drop("status", axis=1)
y = df["status"]

pipeline = build_pipeline()
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(pipeline, "modelo_match_decision.joblib")
