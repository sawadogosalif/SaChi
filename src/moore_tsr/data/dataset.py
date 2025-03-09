from typing import Tuple
import pandas as pd
from .cache import save_to_cache, load_from_cache

def load_split_data(
    dataset_name: str = "sawadogosalif/MooreFRCollections",
    train_size: float = 0.8,
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Charge et divise le dataset en ensembles d'entraînement, de validation et de test.
    Utilise un cache pour éviter de retraiter si déjà fait.
    """
    
    cached_data = load_from_cache()
    if cached_data:
        return cached_data 

    logger.info("Chargement du dataset depuis Hugging Face")
    dataset = load_dataset(dataset_name, split="train")
    df = dataset.to_pandas()

    logger.info("Mélange des données")
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # Division des données
    n = len(df)
    train_end = int(n * train_size)
    test_end = train_end + int(n * test_size)

    logger.info("Ajout d'une colonne 'split'")
    df["split"] = "train"
    df.loc[train_end:test_end, "split"] = "test"
    df.loc[test_end:, "split"] = "val"

    logger.info("Prétraitement des textes")
    df["french"] = df["french"].parallel_apply(preprocess_text)
    df["moore"] = df["moore"].parallel_apply(preprocess_text)

    # Création des sous-ensembles
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    save_to_cache(train_df, val_df, test_df)

    return train_df, val_df, test_df



def get_batch_pairs(
    df: pd.DataFrame,
    batch_size: int,
    src_lang: str = "french",
    tgt_lang: str = "moore",
) -> Tuple[list, list]:
    """
    Génère un batch de paires de traduction.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données
        batch_size (int): Taille du batch
        src_lang (str): Langue source
        tgt_lang (str): Langue cible
    
    Returns:
        Tuple[list, list]: Paires de textes (source, cible)
    """
    batch = df.sample(batch_size, random_state=42)
    src_texts = batch[src_lang].tolist()
    tgt_texts = batch[tgt_lang].tolist()
    return src_texts, tgt_texts
