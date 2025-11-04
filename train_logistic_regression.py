import pandas as pd
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import GridSearchCV, cross_val_predict, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score

from src.model_utils import prepare_data

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

results_df = pd.DataFrame(results).sort_values(by='best_balanced_accuracy_cv', ascending=False)

results_df_readable = results_df.rename(columns={
    'model__C': 'C',
    'model__penalty': 'penalty',
    'model__solver': 'solver',
    'pca__n_components': 'pca_components'
})

results_df_readable['best_balanced_accuracy_cv'] = results_df_readable['best_balanced_accuracy_cv'].round(4)
results_df_readable['refit_time'] = results_df_readable['refit_time'].round(4)

print("\nWyniki GridSearch:")
print(results_df_readable)

output_path = 'graphs/grid_search_results.csv'
results_df_readable.to_csv(output_path, index=False, encoding='utf-8-sig')

best_config = results_df.iloc[0]
print("\nNajlepsza znaleziona konfiguracja ")
print(best_config)

best_data_df = data_variants[best_config['data_variant']]
X_train_best, X_test_best, y_train_best, y_test_best = prepare_data(best_data_df, best_config['feature_set'])

# trening dla najlepszych parametrów
final_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=int(best_config['pca__n_components']))),
    ('model', LogisticRegression(
        C=best_config['model__C'],
        penalty=best_config['model__penalty'],
        solver=best_config['model__solver'],
        max_iter=3000
    ))
])
final_pipeline.fit(X_train_best, y_train_best)

# badanie najlepszego progu prawdopodobieństwa do klasyfikacji
tscv = TimeSeriesSplit(n_splits=5)
# prawdopodobieństwa, że wygrają gospodarze
y_train_pred_proba = cross_val_predict(final_pipeline, X_train_best, y_train_best, cv=tscv, method='predict_proba')[:, 1]
thresholds_to_check = np.linspace(0, 1, 201)
ba_scores = []
for threshold in thresholds_to_check:
    y_pred_loop = (y_train_pred_proba >= threshold).astype(int)
    score = balanced_accuracy_score(y_train_best, y_pred_loop)
    ba_scores.append(score)

best_ba_index = np.argmax(ba_scores)
best_threshold = thresholds_to_check[best_ba_index]
max_ba = ba_scores[best_ba_index]

print(f"Najwyższy osiągnięty Balanced Accuracy na zbiorze treningowym (CV): {max_ba:.4f}")
print(f"Najlepszy próg decyzyjny maksymalizujący Balanced Accuracy: {best_threshold:.4f}")

y_pred_proba_test = final_pipeline.predict_proba(X_test_best)[:, 1]
y_pred_final = (y_pred_proba_test >= best_threshold).astype(int)

print("\n RAPORT KLASYFIKACJI (z progiem prawdopodobieństwa = {:.4f}) ".format(best_threshold))
print(classification_report(y_test_best, y_pred_final))

conf_matrix = confusion_matrix(y_test_best, y_pred_final)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=['0 - wygrana gości', '1 - wygrana gospodarzy'],
            yticklabels=['0 - wygrana gości', '1 - wygrana gospodarzy'])
plt.xlabel("Przewidywane klasy")
plt.ylabel("Rzeczywiste klasy")
plt.title("Macierz pomyłek dla regresji logistycznej")
plt.savefig("graphs/confusion_matrix_lr.png", dpi=300, bbox_inches='tight')
plt.show()

with open('models/LR_model.pkl', 'wb') as file:
    pickle.dump(final_pipeline, file)