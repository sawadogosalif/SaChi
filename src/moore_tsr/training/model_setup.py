from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import Tuple
from loguru import logger

"""Note 1 :  NVIDIA recommande que les tailles des tenseurs (comme l’embedding layer) soient des multiples de **8, 16 ou 32** pour maximiser l'utilisation des **Tensor Cores**, qui accélèrent les calculs de multiplication matricielle en virgule flottante. C’est documenté ici :  
🔗 [NVIDIA Deep Learning Performance Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc)
Les Tensor Cores sont activés **uniquement** si les dimensions des matrices respectent certains critères :
- **FP16 et BF16** : multiples de **8**
- **INT8** : multiples de **16**
- **FP32** : pas de contrainte mais moins optimisé  

Si la taille du vocabulaire du modèle n'est **pas un multiple de 8**, les calculs basculent en mode non optimisé, ce qui peut entraîner un ralentissement.
Note 2: la library transformers a beaucoup changé
"""
def setup_model_and_tokenizer(
    model_name: str = "facebook/nllb-200-distilled-600M",
    new_lang_code: str = "moore_open",
    device: str = "cuda",
) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
    """
    Initialise le modèle et le tokenizer, et ajoute un nouveau token de langue.

    Args:
        model_name (str): Nom du modèle pré-entraîné.
        new_lang_code (str): Code de la nouvelle langue à ajouter.
        device (str): Device à utiliser ("cuda" ou "cpu").

    Returns:
        Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]: Modèle et tokenizer configurés.
    """
    # Chargement du tokenizer
    logger.info(f"Model name {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info(f"tokenize loaded and leen is {len(tokenizer)}.")

    # Ajout du nouveau token de langue
    tokenizer.add_special_tokens({"additional_special_tokens": [new_lang_code]})
    
    # Chargement du modèle
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    logger.info(f"Model loaded.")
    
    new_vocab_size = (len(tokenizer) + 7) // 8 * 8  # Arrondi au multiple de 8 : Voir Note 1
    model.resize_token_embeddings(new_vocab_size)
    logger.info(f"new token length {len(tokenizer)}")

    model.to(device)


    return model, tokenizer
