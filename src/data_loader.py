import pandas as pd

def load_full_data(path):
    return pd.read_csv(path)

def get_test_data(full_df):
    return full_df[full_df['cycle_phase'] == 'test_output_5'].copy()

def filter_by_time(test_df, day_week, hour):
    """
    Retourne la ligne du jeu de test correspondant exactement au jour et a l'heure.
    Si aucun resultat exact, prend la ligne avec l'heure la plus proche.
    """
    mask = (test_df['day_week'] == day_week) & (test_df['hour'] == hour)
    if mask.any():
        return test_df[mask].iloc[0]
    else:
        # Calcul de la difference d'heure et prise de l'indice minimal
        idx = (test_df['hour'] - hour).abs().idxmin()
        return test_df.loc[idx]
