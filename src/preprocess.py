from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

def build_pipeline():
    # Colunas que existem no DataFrame
    cat_features = [
        "nivel_academico", "nivel_ingles", "nivel_espanhol",
        "area_atuacao", "nivel_profissional"
    ]

    text_features = ["cv_texto", "atividades", "competencias"]

    # Transformações
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    text_transformer = Pipeline(steps=[
        ("tfidf", TfidfVectorizer(max_features=1000))
    ])

    # ColumnTransformer adaptado ao seu DataFrame
    preprocessor = ColumnTransformer(transformers=[
        ("cat", categorical_transformer, cat_features),
        ("text_cv", text_transformer, "cv_texto"),
        ("text_atividades", text_transformer, "atividades"),
        ("text_competencias", text_transformer, "competencias"),
        # adicione colunas numéricas aqui se existirem no futuro
    ])

    # Pipeline final
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
    ])

    return pipeline
