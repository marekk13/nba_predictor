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