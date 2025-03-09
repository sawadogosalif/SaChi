from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import Tuple
from loguru import logger

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
    logger.info(f"tokenize loaded.")

    # Ajout du nouveau token de langue
    old_len = len(tokenizer) - int(new_lang_code in tokenizer.added_tokens_encoder)
    tokenizer.lang_code_to_id[new_lang_code] = old_len - 1
    tokenizer.id_to_lang_code[old_len - 1] = new_lang_code
    
    # Chargement du modèle
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    logger.info(f"Model loaded.")

    model.resize_token_embeddings(len(tokenizer))
    model.to(device)


    return model, tokenizer
