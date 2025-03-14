import pandas as pd
import numpy as np
from loguru import logger
from typing import List, Dict
import plotly.express as px
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


def create_loss_visualization(train_losses, val_losses, step_markers, save_path, drive_save_path=None):
    """Create and save visualization of training and validation losses."""
    # Create DataFrame for plotting
    train_df = pd.DataFrame({
        "Step": range(len(train_losses)),
        "Loss": train_losses,
        "Type": "Training"
    })
    
    val_df = pd.DataFrame({
        "Step": step_markers,
        "Loss": val_losses,
        "Type": "Validation"
    })
    
    df_combined = pd.concat([train_df, val_df], ignore_index=True)
    
    # Create plot
    fig = px.line(
        df_combined, 
        x="Step", 
        y="Loss", 
        color="Type",
        title="Training and Validation Loss",
        template="plotly_white"
    )
    
    # Add smoothed training loss line
    if len(train_losses) > 100:
        window_size = min(100, len(train_losses) // 10)
        train_smooth = np.convolve(train_losses, np.ones(window_size)/window_size, mode='valid')
        smooth_steps = range(window_size-1, len(train_losses))
        
        fig.add_scatter(
            x=smooth_steps, 
            y=train_smooth, 
            mode='lines',
            line=dict(color='rgba(0,0,255,0.5)', width=2),
            name='Training (Smoothed)'
        )
    
    # Save plot
    plot_path = f"{save_path}/training_validation_loss.html"
    fig.write_html(plot_path)
    logger.info(f"Loss visualization saved to {plot_path}")
    
    # Save to Google Drive if mounted
    if drive_save_path:
        drive_plot_path = f"{drive_save_path}/training_validation_loss.html"
        fig.write_html(drive_plot_path)
        logger.info(f"Loss visualization also saved to {drive_plot_path}")
