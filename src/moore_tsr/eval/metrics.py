from typing import Dict, List
import pandas as pd
from loguru import logger
from sacrebleu import corpus_bleu
from tqdm.auto import tqdm
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from moore_tsr.data.preprocessing import preprocess_text
from moore_tsr.utils.helpers import get_batch_pairs, cleanup

def calculate_bleu_scores(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    df: pd.DataFrame,
    src_lang: str = "fra_Latn",
    tgt_lang: str = "moore_open",
    src_field: str = "french",
    tgt_field: str = "moore",
    batch_size: int = 16,
) -> Dict[str, float]:
    """
    Calcule le score BLEU pour un ensemble de données donné.

    Args:
        model (AutoModelForSeq2SeqLM): Modèle de traduction.
        tokenizer (AutoTokenizer): Tokenizer configuré.
        df (pd.DataFrame): DataFrame contenant les données de test.
        src_lang (str): Code de la langue source.
        tgt_lang (str): Code de la langue cible.
        src_field (str): Nom de la colonne source dans le DataFrame.
        tgt_field (str): Nom de la colonne cible dans le DataFrame.
        batch_size (int): Taille du batch pour l'inférence.

    Returns:
        Dict[str, float]: Score BLEU et détails.
    """
    model.eval()  # Passage en mode évaluation

    # Préparation des données
    references = []
    translations = []

    # Traduction par batch
    for i in tqdm(range(0, len(df), batch_size), desc="Calcul du score BLEU"):
        batch = df.iloc[i:i + batch_size]
        src_texts = batch[src_field].apply(preprocess_text).tolist()
        ref_texts = batch[tgt_field].apply(preprocess_text).tolist()

        # Génération des traductions
        inputs = tokenizer(src_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model.generate(
                **inputs.to(model.device),
                forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
                max_length=128,
            )
        translated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        references.extend(ref_texts)
        translations.extend(translated_texts)

    # Calcul du score BLEU
    bleu_score = corpus_bleu(translations, [references]).score

    return {
        "bleu_score": bleu_score,
        "direction": f"{src_lang} → {tgt_lang}",
        "num_samples": len(df),
    }

def evaluate_model_with_bleu(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    df_test: pd.DataFrame,
) -> Dict[str, float]:
    """
    Évalue le modèle dans les deux directions de traduction (français → mooré et mooré → français).

    Args:
        model (AutoModelForSeq2SeqLM): Modèle à évaluer.
        tokenizer (AutoTokenizer): Tokenizer configuré.
        df_test (pd.DataFrame): Données de test.

    Returns:
        Dict[str, float]: Résultats des scores BLEU dans les deux directions.
    """
    # Évaluation français → mooré
    fr_to_moore = calculate_bleu_scores(
        model, tokenizer, df_test,
        src_lang="fra_Latn", tgt_lang="moore_open",
        src_field="french", tgt_field="moore"
    )

    # Évaluation mooré → français
    moore_to_fr = calculate_bleu_scores(
        model, tokenizer, df_test,
        src_lang="moore_open", tgt_lang="fra_Latn",
        src_field="moore", tgt_field="french"
    )

    return {
        "fr_to_moore_bleu": fr_to_moore["bleu_score"],
        "moore_to_fr_bleu": moore_to_fr["bleu_score"],
        "average_bleu": (fr_to_moore["bleu_score"] + moore_to_fr["bleu_score"]) / 2,
    }


def evaluate_model_loss(model, tokenizer, eval_df, batch_size=16, max_length=128, device="cuda"):
    """Evaluate model on validation data and return loss."""
    model.eval()
    total_loss = 0.0
    batches = 0
    cleanup()

    with torch.no_grad():
        for i in range(0, len(eval_df), batch_size):
            try:
                src_texts, tgt_texts, src_lang, tgt_lang = get_batch_pairs(
                    eval_df, batch_size
                )
                with torch.cuda.amp.autocast():
                    tokenizer.src_lang = src_lang
                    x = tokenizer(src_texts, return_tensors='pt', padding=True, 
                            truncation=True, max_length=max_length).to(model.device)
                
                    tokenizer.src_lang = tgt_lang
                    y = tokenizer(tgt_texts, return_tensors='pt', padding=True, 
                            truncation=True, max_length=max_length).to(model.device)
                    y.input_ids[y.input_ids == tokenizer.pad_token_id] = -100
                    
                    outputs = model(**x, labels=y.input_ids)
                    loss = outputs.loss
                
                total_loss += loss.item()
                batches += 1
            except RuntimeError as e:
                logger.error(f"Error during evaluation: {e}")
                cleanup()
                continue
    
    model.train()
    return total_loss / max(1, batches)