import pandas as pd
import numpy as np

def analyze_na_outliers(df, context_cols):
    missing_data = pd.isnull(df).mean()*100

    # Wykrywanie grubych błędów
    outliers = {}
    for col in df.select_dtypes(include=['float', 'int']).columns:
        mean = df[col].mean()
        std = df[col].std()
        threshold = 4

        outliers[col] = df.loc[
            (df[col] < mean - threshold * std) | (df[col] > mean + threshold * std),
            context_cols + [col]
        ]

    errors = {}
    for col in df.loc[:, df.columns.str.contains('%')].columns:
        errors[col] = df.loc[
            df[col]<0,
            context_cols + [col]
        ]

    return missing_data, outliers, errors


def extract_date(game_id):
    game_id_str=str(game_id)
    season_prefix = game_id_str[2:4]
    month = int(game_id_str[4:6])
    day = int(game_id_str[6:8])

    if 10 <= month <= 12:
        year = '20'+str(int(season_prefix) - 1)
    else:
        year = '20'+str(int(season_prefix))

    date_str = f"{year}-{month:02d}-{day:02d}"
    return date_str

def add_last_10(df):
    df['last10'] = (
        df.groupby(['team', 'Season'])['win']
        .rolling(window=10, min_periods=1)
        .sum()
        .shift(1)
        .reset_index(level=['team', 'Season'], drop=True)
    )

    first_game_mask = df.groupby(['team', 'Season']).cumcount() == 0
    df.loc[first_game_mask, 'last10'] = 0.0
    return df.sort_index()

def calculate_streak(win_series):
    streaks = []
    streak = 0
    last_win = None

    for i in range(1, len(win_series)):
        win = win_series.iloc[i - 1]  # bierzemy wynik meczu poprzedzającego aktualny
        if win == last_win:
            streak += 1
        else:
            streak = 1
        streaks.append(streak if win == 1 else -streak)
        last_win = win

    # 1 mecz nie może mieć streaka, więc dodajemy wartość 0 na początku
    streaks.insert(0, 0)

    return streaks

def calculate_rolling_stats(df, window, columns, cols_concat):
    df_copy = df.copy()
    results = pd.DataFrame()

    for col in columns:
        if window == 'all':
            rolling_mean = (
                df_copy.groupby(['Season', 'team'])[col]
                .expanding()
                .mean()
                .shift(1)
                .reset_index(level=['Season', 'team'], drop=True)
            )
        else:
            rolling_mean = (
                df_copy.groupby(['Season', 'team'])[col]
                .rolling(window=window, min_periods=1)
                .mean()
                .shift(1)
                .reset_index(level=['Season', 'team'], drop=True)
            )

        results[col] = rolling_mean
    return pd.concat([df_copy[cols_concat].reset_index(drop=True), results], axis=1)


def impute_first_rows(rolling_avgs, team_stats_boxscore, columns):
    first_season = '1718'
    df_copy = team_stats_boxscore.copy()
    rolling_avgs = rolling_avgs.copy()

    # dla każdego sezonu (oprócz pierwszego) używamy ostatnich wartości z poprzedniego sezonu
    for season in df_copy['Season'].unique()[1:]:
        yr1, yr2 = int(season[:2]), int(season[2:])
        prev_season = str(yr1-1)+str(yr2-1)  # poprzedni sezon

        # dla każdej drużyny w bieżącym sezonie
        for team in np.sort(df_copy[df_copy['Season'] == season]['team'].unique()):
            # indeks pierwszego rekordu w bieżącym sezonie
            current_mask = (rolling_avgs['Season'] == season) & (rolling_avgs['team'] == team)

            # ostatnia wartość z poprzedniego sezonu
            prev_mask = (rolling_avgs['Season'] == prev_season) & (rolling_avgs['team'] == team)

            if prev_mask.any():
                prev_values = rolling_avgs[prev_mask].iloc[-1][columns]
                if current_mask.any():
                    first_idx = rolling_avgs[current_mask].index[0]
                    rolling_avgs.loc[first_idx, columns] = prev_values

    # la pierwszego sezonu średnia ligowa
    league_avg = df_copy[df_copy['Season'] == first_season][columns].mean()

    for team in np.sort(df_copy[df_copy['Season'] == first_season]['team'].unique()):
        mask = (rolling_avgs['Season'] == first_season) & (rolling_avgs['team'] == team)
        if mask.any():
            first_idx = rolling_avgs[mask].index[0]
            rolling_avgs.loc[first_idx, columns] = league_avg
    return rolling_avgs