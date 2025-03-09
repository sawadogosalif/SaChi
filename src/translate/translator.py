from typing import List, Optional
import torch
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

def load_translation_pipeline(
    model_path: str,
    device: Optional[str] = None,
) -> tp.Tuple[AutoModelForSeq2SeqLM, NllbTokenizer]:
    """
    Charge le modèle et le tokenizer pour la traduction.

    Args:
        model_path (str): Chemin vers le modèle fine-tuné.
        device (str, optional): Device à utiliser ("cuda" ou "cpu"). Par défaut, détecte automatiquement.

    Returns:
        Tuple[AutoModelForSeq2SeqLM, NllbTokenizer]: Modèle et tokenizer chargés.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Chargement du tokenizer
    tokenizer = NllbTokenizer.from_pretrained(model_path)
    
    # Chargement du modèle
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    return model, tokenizer

def translate_text(
    model: AutoModelForSeq2SeqLM,
    tokenizer: NllbTokenizer,
    texts: List[str],
    src_lang: str = "fra_Latn",
    tgt_lang: str = "moore_open",
    max_length: int = 128,
    num_beams: int = 5,
    no_repeat_ngram_size: int = 3,
) -> List[str]:
    """
    Traduit une liste de textes de la langue source à la langue cible.

    Args:
        model (AutoModelForSeq2SeqLM): Modèle de traduction.
        tokenizer (NllbTokenizer): Tokenizer configuré.
        texts (List[str]): Textes à traduire.
        src_lang (str): Code de la langue source.
        tgt_lang (str): Code de la langue cible.
        max_length (int): Longueur maximale des séquences générées.
        num_beams (int): Nombre de beams pour la recherche.
        no_repeat_ngram_size (int): Taille des n-grammes à éviter.

    Returns:
        List[str]: Textes traduits.
    """
    # Tokenisation des entrées
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(model.device)
    
    # Génération des traductions
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
            max_length=max_length,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
    
    # Décodage des sorties
    translated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return translated_texts