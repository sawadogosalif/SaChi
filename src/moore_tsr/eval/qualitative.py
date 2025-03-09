import pandas as pd
from typing import List, Dict
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer
from moore_tsr.data.preprocessing import preprocess_text

def evaluate_qualitative(
    model: AutoModelForSeq2SeqLM,
    tokenizer: NllbTokenizer,
    df: pd.DataFrame,
    num_examples: int = 5,
    src_lang: str = "fra_Latn",
    tgt_lang: str = "moore_open",
    src_field: str = "french",
    tgt_field: str = "moore",
) -> List[Dict[str, str]]:
    """
    Génère des exemples de traductions pour une évaluation qualitative.

    Args:
        model (AutoModelForSeq2SeqLM): Modèle de traduction.
        tokenizer (NllbTokenizer): Tokenizer configuré.
        df (pd.DataFrame): DataFrame contenant les données.
        num_examples (int): Nombre d'exemples à générer.
        src_lang (str): Code de la langue source.
        tgt_lang (str): Code de la langue cible.
        src_field (str): Nom de la colonne source dans le DataFrame.
        tgt_field (str): Nom de la colonne cible dans le DataFrame.

    Returns:
        List[Dict[str, str]]: Exemples de traductions.
    """
    results = []
    sample = df.sample(num_examples, random_state=42)

    for _, row in sample.iterrows():
        src_text = preprocess_text(row[src_field])
        ref_text = preprocess_text(row[tgt_field])

        # Traduction
        inputs = tokenizer(src_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model.generate(
                **inputs.to(model.device),
                forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
                max_length=128,
            )
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        results.append({
            "source": src_text,
            "reference": ref_text,
            "translation": translated_text,
            "direction": f"{src_lang} → {tgt_lang}",
        })

    return results

def display_translation_examples(examples: List[Dict[str, str]]):
    """
    Affiche des exemples de traductions de manière lisible.

    Args:
        examples (List[Dict[str, str]]): Liste des exemples de traductions.
    """
    for i, example in enumerate(examples):
        print(f"\n=== Exemple {i + 1} ===")
        print(f"Source ({example['direction']}): {example['source']}")
        print(f"Référence: {example['reference']}")
        print(f"Traduction: {example['translation']}")