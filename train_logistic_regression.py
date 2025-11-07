import pandas as pd
import sqlite3
import numpy as np

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from src.model_utils import (
    prepare_data,
    process_search_results,
    save_config_to_json,
    find_best_threshold,
    evaluate_and_save_artifacts
)

DB_PATH = 'data/transformed/team_moving_avgs_merged.sqlite'
pd.set_option('display.max_columns', 20)

con = sqlite3.connect(DB_PATH)
data_variants = {
    'team_last_20': pd.read_sql_query("SELECT * FROM team_last_20", con),
    'team_last_30': pd.read_sql_query("SELECT * FROM team_last_30", con),
    'team_last_40': pd.read_sql_query("SELECT * FROM team_last_40", con),
    'team_all_season': pd.read_sql_query("SELECT * FROM team_all_season", con),
}
con.close()

for name, df in data_variants.items():
    df['Date'] = pd.to_datetime(df['Date'])

results = []
feature_sets = {'all': 'Wszystkie cechy', 'no_totals': 'Bez cech `totals`', 'no_prc': 'Bez cech `prc`'}

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('model', LogisticRegression(max_iter=3000))
])

param_grid = {
    'pca__n_components': [5, 10, 15, None],  # None - bez PCA
    'model__C': np.linspace(0.0, 2.0, 9)[1:],
    'model__penalty': ['l1', 'l2'],
    'model__solver': ['liblinear', 'saga']
}

# badanie przestrzeni parametrów i wariantów danych
for data_name, data_df in data_variants.items():
    for fs_name, fs_label in feature_sets.items():
        print(f"Testowanie: [Dane: {data_name}] | [Cechy: {fs_label}]")

        X_train, X_test, y_train, y_test = prepare_data(data_df, feature_set=fs_name)

        tscv = TimeSeriesSplit(n_splits=5)
        grid_search = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='balanced_accuracy', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)

        results.append({
            'data_variant': data_name,
            'feature_set': fs_name,
            'best_balanced_accuracy_cv': grid_search.best_score_,
            'refit_time': grid_search.refit_time_,
            **grid_search.best_params_
        })

best_config = process_search_results(results, 'graphs/lr_grid_search_results.csv')
save_config_to_json(best_config, 'models/lr_best_config.json')

# trening i ewaluacja modelu na najlepszych znalezionych parametrach
best_data_df = data_variants[best_config['data_variant']]
X_train_best, X_test_best, y_train_best, y_test_best = prepare_data(best_data_df, best_config['feature_set'])

n_components_value = best_config.get('pca__n_components')
if n_components_value is not None:
    n_components_value = int(n_components_value)

final_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=PCA(n_components=n_components_value))),
    ('model', LogisticRegression(
        C=best_config['model__C'], penalty=best_config['model__penalty'],
        solver=best_config['model__solver'], max_iter=3000
    ))
])
final_pipeline.fit(X_train_best, y_train_best)

best_threshold = find_best_threshold(final_pipeline, X_train_best, y_train_best)
evaluate_and_save_artifacts(
    pipeline=final_pipeline, X_test=X_test_best, y_test=y_test_best,
    threshold=best_threshold, model_name="LR",
    class_names=['Wygrana gości (0)', 'Wygrana gospodarzy (1)']
)