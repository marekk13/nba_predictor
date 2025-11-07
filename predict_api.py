import pandas as pd
import sqlite3
import pickle
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import date
from pathlib import Path

from src.model_utils import create_model

app = FastAPI(
    title="NBA Match Outcome Predictor API",
    description="Statyczne API do odtwarzania predykcji dla meczów NBA z lat 2017-2024 na podstawie wytrenowanych modeli.",
    version="1.1.0"
)

MODELS_DIR = Path("models")
DB_PATH = Path("data/transformed/team_moving_avgs_merged.sqlite")

app_state = {}


class PredictionRequest(BaseModel):
    home_team_abbr: str = Field(..., description="Skrót drużyny gospodarzy (np. 'GSW')", example="GSW")
    away_team_abbr: str = Field(..., description="Skrót drużyny gości (np. 'HOU')", example="HOU")
    game_date: date = Field(..., description="Data meczu w formacie YYYY-MM-DD", example="2019-01-03")


class PredictionResponse(BaseModel):
    home_team: str
    away_team: str
    game_date: date
    prediction: str
    win_probability_home: float = Field(..., description="Prawdopodobieństwo wygranej gospodarzy")
    probability_threshold: float = Field(..., description="Optymalny próg prawdopodobieństwa")
    model_used: str = Field(..., description="Model użyty do predykcji (np. 'LR' lub 'NN')")


@app.on_event("startup")
def load_artifacts_and_data():
    print("Rozpoczynanie ładowania modeli i danych...")

    with open(MODELS_DIR / "lr_best_config.json") as f:
        lr_config = json.load(f)
    with open(MODELS_DIR / "nn_best_config.json") as f:
        nn_config = json.load(f)

    app_state["LR"] = {"config": lr_config}
    app_state["NN"] = {"config": nn_config}

    lr_score = lr_config.get('best_balanced_accuracy_cv', 0)
    nn_score = nn_config.get('best_balanced_accuracy_cv', 0)

    app_state['best_model_name'] = 'LR' if lr_score >= nn_score else 'NN'
    print(
        f"Najlepszy model na podstawie walidacji krzyżowej: {app_state['best_model_name']} (LR: {lr_score:.4f}, NN: {nn_score:.4f})")

    required_tables = {
        config['data_variant'] for model_name, data in app_state.items() if 'config' in data
        for config in [data['config']]
    }

    con = sqlite3.connect(DB_PATH)
    app_state['data_variants'] = {}
    for table_name in required_tables:
        print(f"Wczytywanie wariantu danych: {table_name}")
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", con)
        df['Date'] = pd.to_datetime(df['Date'])
        app_state['data_variants'][table_name] = df
    con.close()

    for model_name in ["LR", "NN"]:
        with open(MODELS_DIR / f"{model_name}_model.pkl", 'rb') as f:
            app_state[model_name]['model'] = pickle.load(f)
        with open(MODELS_DIR / f"{model_name}_threshold.pkl", 'rb') as f:
            app_state[model_name]['threshold'] = pickle.load(f)
        with open(MODELS_DIR / f"{model_name}_feature_names.pkl", 'rb') as f:
            app_state[model_name]['features'] = pickle.load(f)
        print(f"Artefakty dla modelu {model_name} załadowane.")



@app.post("/predict", response_model=PredictionResponse)
def predict_outcome(request: PredictionRequest):
    """
    Znajduje mecz i zwraca predykcję z najlepszego wytrenowanego modelu.
    """
    model_name = app_state['best_model_name']
    config = app_state[model_name]['config']
    model_pipeline = app_state[model_name]['model']
    threshold = app_state[model_name]['threshold']
    feature_names = app_state[model_name]['features']
    data_variant_name = config['data_variant']

    source_df = app_state['data_variants'][data_variant_name]

    game_date_ts = pd.Timestamp(request.game_date)

    match_row_df = source_df[
        (source_df['Date'] == game_date_ts) &
        (source_df['home_team'] == request.home_team_abbr) &
        (source_df['away_team'] == request.away_team_abbr)
        ]

    if match_row_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Nie znaleziono meczu dla podanych drużyn i daty w wariancie danych '{data_variant_name}'."
        )

    feature_vector = match_row_df[feature_names]

    if feature_vector.isnull().values.any():
        raise HTTPException(
            status_code=422,
            detail=f"Znaleziono brakujące dane dla tego meczu. Nie można wykonać predykcji."
        )

    win_probability = model_pipeline.predict_proba(feature_vector)[0][1]
    prediction_code = 1 if win_probability >= threshold else 0
    prediction_text = "Wygrana gospodarzy" if prediction_code == 1 else "Wygrana gości"

    return PredictionResponse(
        home_team=request.home_team_abbr,
        away_team=request.away_team_abbr,
        game_date=request.game_date,
        prediction=prediction_text,
        win_probability_home=round(win_probability, 4),
        probability_threshold = round(threshold, 4),
        model_used=model_name
    )