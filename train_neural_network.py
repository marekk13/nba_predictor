import sqlite3
import warnings

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam, AdamW, RMSprop
from tensorflow.keras.callbacks import EarlyStopping

from src.model_utils import (
    prepare_data,
    process_search_results,
    save_config_to_json,
    find_best_threshold,
    evaluate_and_save_artifacts,
    create_model
)

np.random.seed(42)
tf.random.set_seed(42)
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

DB_PATH = 'data/transformed/team_moving_avgs_merged.sqlite'
pd.set_option('display.max_columns', 20)

con = sqlite3.connect(DB_PATH)
data_variants = {
    'team_last_20': pd.read_sql_query("SELECT * FROM team_last_20", con),
    'team_all_season': pd.read_sql_query("SELECT * FROM team_all_season", con),
    # 'team_last_30': pd.read_sql_query("SELECT * FROM team_last_30", con),
    # 'team_last_40': pd.read_sql_query("SELECT * FROM team_last_40", con),
}
con.close()

for name, df in data_variants.items():
    df['Date'] = pd.to_datetime(df['Date'])

feature_sets = {
    'all': 'Wszystkie cechy',
    'no_totals': 'Bez cech `totals`',
}


keras_clf = KerasClassifier(
    model=create_model,
    epochs=50,
    batch_size=32,
    verbose=0,
    hidden_layer1_size=40, hidden_layer1_activation='relu', dropout1_rate=0.2,
    hidden_layer2_size=20, hidden_layer2_activation='relu', dropout2_rate=0.2,
    hidden_layer3_size=0, hidden_layer3_activation='relu', dropout3_rate=0.2,
    learning_rate=0.001,
    optimizer_name='adam'
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('model', keras_clf)
])

param_distributions = {
    'pca__n_components': [5, 20, None],
    'model__hidden_layer1_size': [30, 40, 50, 60],
    'model__hidden_layer1_activation': ['relu', 'tanh'],
    'model__dropout1_rate': [0.1, 0.2, 0.3],
    'model__hidden_layer2_size': [15, 20, 25, 30],
    'model__hidden_layer2_activation': ['relu', 'tanh'],
    'model__dropout2_rate': [0.1, 0.2, 0.3],
    'model__learning_rate': [0.01, 0.005, 0.001],

    'model__batch_size': [32, 64, 128],
    'model__optimizer_name': ['adam', 'adamw', 'rmsprop'],
    'model__hidden_layer3_size': [0, 10, 15, 20],
    'model__hidden_layer3_activation': ['relu', 'tanh'],
    'model__dropout3_rate': [0.1, 0.2]
}

early_stopping = EarlyStopping(
    monitor='val_loss',  # sprawdzana strata na zbiorze walidacyjnym
    patience=5,          # liczba epok bez poprawy, po której trening zostanie przerwany
    restore_best_weights=True # przywróć wagi z najlepszej epoki
)

results = []
tscv = TimeSeriesSplit(n_splits=5)

random_search = RandomizedSearchCV(
    pipeline,
    param_distributions,
    n_iter=40,
    cv=tscv,
    scoring='balanced_accuracy',
    verbose=1,
    n_jobs=-1,
    random_state=42
)

for data_name, data_df in data_variants.items():
    for fs_name, fs_label in feature_sets.items():
        print(f"Testowanie: [Dane: {data_name}] | [Cechy: {fs_label}]")

        X_train, X_test, y_train, y_test = prepare_data(data_df, feature_set=fs_name)

        random_search.fit(
            X_train,
            y_train,
            model__callbacks=[early_stopping]
        )

        results.append({
            'data_variant': data_name,
            'feature_set': fs_name,
            'best_balanced_accuracy_cv': random_search.best_score_,
            'refit_time': random_search.refit_time_,
            **random_search.best_params_
        })

best_config = process_search_results(results, 'graphs/nn_random_search_results.csv')
save_config_to_json(best_config, 'models/nn_best_config.json')

# trening i ewaluacja modelu na najlepszych znalezionych parametrach
best_data_df = data_variants[best_config['data_variant']]
X_train_best, X_test_best, y_train_best, y_test_best = prepare_data(best_data_df, best_config['feature_set'])

final_keras_clf = KerasClassifier(
    model=create_model,
    epochs=100,
    verbose=0,
    batch_size=best_config['model__batch_size'],
    hidden_layer1_size=best_config['model__hidden_layer1_size'],
    hidden_layer1_activation=best_config['model__hidden_layer1_activation'],
    dropout1_rate=best_config['model__dropout1_rate'],
    hidden_layer2_size=best_config['model__hidden_layer2_size'],
    hidden_layer2_activation=best_config['model__hidden_layer2_activation'],
    dropout2_rate=best_config['model__dropout2_rate'],
    learning_rate=best_config['model__learning_rate'],
    optimizer_name=best_config['model__optimizer_name'],
    hidden_layer3_size=best_config.get('model__hidden_layer3_size', 0),
    hidden_layer3_activation=best_config.get('model__hidden_layer3_activation', 'relu'),
    dropout3_rate=best_config.get('model__dropout3_rate', 0.2)
)

final_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=best_config.get('pca__n_components'))),
    ('model', final_keras_clf)
])
final_pipeline.fit(X_train_best, y_train_best, model__callbacks=[early_stopping])

best_threshold = find_best_threshold(final_pipeline, X_train_best, y_train_best)
evaluate_and_save_artifacts(
    pipeline=final_pipeline, X_test=X_test_best, y_test=y_test_best,
    threshold=best_threshold, model_name="NN",
    class_names=['Wygrana gości (0)', 'Wygrana gospodarzy (1)']
)