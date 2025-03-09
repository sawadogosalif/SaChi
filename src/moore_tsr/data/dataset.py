import pandas as pd
import swifter
swifter.set_defaults(force_parallel=True)

from datasets import load_dataset
from typing import Tuple
from .preprocessing import preprocess_text

def load_split_data(
    dataset_name: str = "sawadogosalif/MooreFRCollections",
    train_size: float = 0.8,
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Charge et divise le dataset en ensembles d'entraînement, de validation et de test.
    
    Args:
        dataset_name (str): Nom du dataset sur Hugging Face
        train_size (float): Proportion des données d'entraînement
        test_size (float): Proportion des données de test
        val_size (float): Proportion des données de validation
        random_seed (int): Graine aléatoire pour la reproductibilité
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: DataFrames pour train, val, test
    """
    logger.info("Chargement du dataset")
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
    df["french"] = df["french"].swifter.apply(preprocess_text)
    df["moore"] = df["moore"].swifter.apply(preprocess_text)
    
    # Création des sous-ensembles
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()
    
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
