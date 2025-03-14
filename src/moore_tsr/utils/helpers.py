import gc
import torch
import random
import numpy as np
from loguru import logger 
from transformers import PreTrainedModel, PreTrainedTokenizer

def cleanup():
    """
    Libère la mémoire GPU et nettoie le cache.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def set_random_seed(seed: int = 2025):
    """
    Définit la seed aléatoire pour assurer la reproductibilité.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def format_time(seconds: float) -> str:
    """
    Formate une durée en secondes en une chaîne lisible.

    Args:
        seconds (float): Durée en secondes.

    Returns:
        str: Durée formatée (ex: "2h 30m 15s").
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours}h {minutes}m {seconds}s"

def get_batch_pairs(data, batch_size):
    """Translation bidirectionelle  
    Sélectionne un couple de langues au hasard et retourne un batch de textes source et cible.
    """
    
    language_pairs = [('french', "fra_Latn"), ('moore', "moor_Latn")]
    (src_lang, src_lang_code), (tgt_lang, tgt_lang_code) = random.sample(language_pairs, 2)

    src_texts, tgt_texts = [], []
    for _ in range(batch_size):
        item = data.iloc[random.randint(0, len(data) - 1)]
        src_texts.append(item[src_lang])  # Texte source
        tgt_texts.append(item[tgt_lang])  # Texte cible

    return src_texts, tgt_texts, src_lang_code, tgt_lang_code

    
def cleanup():
    """Clear memory to prevent CUDA out-of-memory errors."""
    gc.collect()
    torch.cuda.empty_cache()

def save_model_and_tokenizer(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, path: str, drive_path: str = None):
    """
    Sauvegarde le modèle et le tokenizer à l'emplacement spécifié.
    Si un chemin Google Drive est fourni, il sauvegarde aussi une copie sur Google Drive.
    """
    logger.info(f"Saving model to {path}")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    
    if drive_path:
        logger.info(f"Saving model to Google Drive at {drive_path}")
        model.save_pretrained(drive_path)
        tokenizer.save_pretrained(drive_path)
