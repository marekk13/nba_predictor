import json
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam, AdamW, RMSprop


THRESHOLD_DATE = '2022-10-01'
TARGET_COLUMN = 'home_win'

totals_columns = ['home_FG', 'home_FGA', 'home_FG%', 'home_3P', 'home_3PA',
                  'home_3P%', 'home_FT', 'home_FTA', 'home_FT%', 'home_ORB', 'home_DRB',
                  'home_TRB', 'home_AST', 'home_STL', 'home_BLK', 'home_TOV', 'home_PF',
                  'home_PTS', 'home_TS%', 'away_FG', 'away_FGA',
                  'away_FG%', 'away_3P', 'away_3PA', 'away_3P%', 'away_FT', 'away_FTA',
                  'away_FT%', 'away_ORB', 'away_DRB', 'away_TRB', 'away_AST', 'away_STL',
                  'away_BLK', 'away_TOV', 'away_PF', 'away_PTS']

prc_columns = ['home_eFG%', 'home_3PAr', 'home_FTr', 'home_ORB%', 'home_DRB%',
               'home_TRB%', 'home_AST%', 'home_STL%', 'home_BLK%', 'home_TOV%',
               'home_ORtg', 'home_DRtg', 'away_TS%', 'away_eFG%', 'away_3PAr', 'away_FTr',
               'away_ORB%', 'away_DRB%', 'away_TRB%', 'away_AST%', 'away_STL%',
               'away_BLK%', 'away_TOV%', 'away_ORtg', 'away_DRtg']

def prepare_data(data, feature_set='all'):
    """Przygotowuje dane, dzieląc je na zbiory treningowe i testowe."""
    data = data.dropna(how='any')

    cols_to_drop = ['game_id', 'Date', 'Season', 'home_team', 'away_team', TARGET_COLUMN]

    if feature_set == 'no_totals':
        valid_totals = [col for col in totals_columns if col in data.columns]
        cols_to_drop.extend(valid_totals)
    elif feature_set == 'no_prc':
        valid_prc = [col for col in prc_columns if col in data.columns]
        cols_to_drop.extend(valid_prc)

    X = data.drop(columns=cols_to_drop)
    y = data[TARGET_COLUMN]

    train_mask = data['Date'] < THRESHOLD_DATE
    test_mask = data['Date'] >= THRESHOLD_DATE

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    return X_train, X_test, y_train, y_test


def process_search_results(results, output_path):
    """Przetwarza wyniki z GridSearchCV/RandomizedSearchCV, zapisuje do CSV i zwraca najlepszą konfigurację."""
    results_df = pd.DataFrame(results).sort_values(by='best_balanced_accuracy_cv', ascending=False)

    renamed_columns = {col: col.replace('model__', '').replace('pca__', '') for col in results_df.columns}
    results_df_readable = results_df.rename(columns=renamed_columns)

    results_df_readable['best_balanced_accuracy_cv'] = results_df_readable['best_balanced_accuracy_cv'].round(4)
    results_df_readable['refit_time'] = results_df_readable['refit_time'].round(4)

    print("\nWyniki przeszukiwania hiperparametrów:")
    print(results_df_readable)
    results_df_readable.to_csv(output_path, index=False, encoding='utf-8-sig')

    best_config = results_df.iloc[0].to_dict()
    print("\nNajlepsza znaleziona konfiguracja:")
    print(best_config)
    return best_config


def save_config_to_json(config, filepath):
    """Zapisuje słownik konfiguracji do pliku JSON, konwertując typy numpy."""
    for key, value in config.items():
        if hasattr(value, 'item'):
            config[key] = value.item()
    if pd.isna(config.get('pca__n_components')):
        config['pca__n_components'] = None

    with open(filepath, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Konfiguracja zapisana do pliku {filepath}")


def find_best_threshold(pipeline, X_train, y_train, cv_splits=5):
    """Znajduje i zwraca optymalny próg decyzyjny używając walidacji krzyżowej."""
    print("\nPoszukiwanie optymalnego progu decyzyjnego...")
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    y_train_pred_proba = np.zeros_like(y_train, dtype=float)

    for train_idx, test_idx in tscv.split(X_train):
        pipeline.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        y_train_pred_proba[test_idx] = pipeline.predict_proba(X_train.iloc[test_idx])[:, 1]

    thresholds = np.linspace(0, 1, 201)
    scores = [balanced_accuracy_score(y_train, (y_train_pred_proba >= t).astype(int)) for t in thresholds]

    best_idx = np.argmax(scores)
    best_threshold = thresholds[best_idx]

    print(f"Najlepszy próg decyzyjny: {best_threshold:.4f} (Balanced Accuracy: {scores[best_idx]:.4f})")
    return best_threshold


def evaluate_and_save_artifacts(pipeline, X_test, y_test, threshold, model_name, class_names):
    """Ocenia finalny model, generuje raport, macierz pomyłek i zapisuje potok do pliku."""
    print(f"\n--- RAPORT KLASYFIKACJI DLA {model_name.upper()} (próg = {threshold:.4f}) ---")

    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    print(classification_report(y_test, y_pred, target_names=class_names))

    cm_path = f"graphs/confusion_matrix_{model_name.lower()}.png"
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Przewidziane zwycięstwa gości (0)", "Przewidziane zwycięstwa gospodarzy (1)"],
                yticklabels=["Rzeczywiste zwycięstwa gospodarzy (1)", "Rzeczywiste zwycięstwa gości (0)"])
    plt.xlabel("Przewidywane klasy")
    plt.ylabel("Rzeczywiste klasy")
    plt.title(f"Macierz pomyłek dla {model_name}")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()

    model_path = f"models/{model_name.upper()}_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"Finalny potok zapisany do pliku: {model_path}")

    threshold_path = f"models/{model_name.upper()}_threshold.pkl"
    with open(threshold_path, 'wb') as f:
        pickle.dump(threshold, f)
    print(f"Próg decyzyjny zapisany do pliku: {threshold_path}")

    feature_names_path = f"models/{model_name.upper()}_feature_names.pkl"
    feature_names = list(X_test.columns)
    with open(feature_names_path, 'wb') as f:
        pickle.dump(feature_names, f)
    print(f"Lista nazw cech zapisana do pliku: {feature_names_path}")


def create_model(hidden_layer1_size=40, hidden_layer1_activation='relu', dropout1_rate=0.2,
                 hidden_layer2_size=20, hidden_layer2_activation='relu', dropout2_rate=0.2,
                 hidden_layer3_size=0, hidden_layer3_activation='relu', dropout3_rate=0.2,
                 learning_rate=0.001, optimizer_name='adam', meta=None):
    n_features_in = meta["n_features_in_"]

    model = Sequential([
        Input(shape=(n_features_in,)),
        Dense(hidden_layer1_size, activation=hidden_layer1_activation),
        Dropout(dropout1_rate),
        Dense(hidden_layer2_size, activation=hidden_layer2_activation),
        Dropout(dropout2_rate)
    ])

    if hidden_layer3_size > 0:
        model.add(Dense(hidden_layer3_size, activation=hidden_layer3_activation))
        model.add(Dropout(dropout3_rate))

    model.add(Dense(1, activation='sigmoid'))

    if optimizer_name == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_name == 'adamw':
        optimizer = AdamW(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    else:
        optimizer = Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model