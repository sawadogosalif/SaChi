import gc
import torch
import random
import numpy as np

def cleanup():
    """
    Libère la mémoire GPU et nettoie le cache.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def set_random_seed(seed: int = 42):
    """
    Définit la seed aléatoire pour assurer la reproductibilité.

    Args:
        seed (int): seed aléatoire.
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