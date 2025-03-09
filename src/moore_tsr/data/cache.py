import os
import joblib
import pandas as pd
from loguru import logger
from typing import Tuple

CACHE_DIR = ".cache"
CACHE_FILE = os.path.join(CACHE_DIR, "dataset.pkl")

def save_to_cache(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """Sauvegarde les DataFrames en cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    joblib.dump((train_df, val_df, test_df), CACHE_FILE)
    logger.info(f"✅ Données sauvegardées dans {CACHE_FILE}")

def load_from_cache() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Charge les DataFrames depuis le cache s'ils existent."""
    if os.path.exists(CACHE_FILE):
        logger.info(f"🔄 Chargement des données depuis {CACHE_FILE}")
        return joblib.load(CACHE_FILE)
    return None
