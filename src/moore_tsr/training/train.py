import pandas as pd
import torch
from torch.optim import AdamW
from transformers import get_scheduler, AutoTokenizer, AutoModelForSeq2SeqLM

from tqdm.auto import tqdm
from moore_tsr.data.dataset import get_batch_pairs

from loguru import logger

def train_model(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 5e-5,
    warmup_steps: int = 1000,
    device: str = "cuda",
    save_path: str = "./models/nllb-moore-finetuned",
) -> None:
    """
    Entraîne le modèle de traduction.

    Args:
        model (AutoModelForSeq2SeqLM): Modèle à entraîner.
        tokenizer (AutoTokenizer): Tokenizer configuré.
        train_df (pd.DataFrame): Données d'entraînement.
        val_df (pd.DataFrame): Données de validation.
        num_epochs (int): Nombre d'époques.
        batch_size (int): Taille du batch.
        learning_rate (float): Taux d'apprentissage.
        warmup_steps (int): Nombre de pas de warmup.
        device (str): Device à utiliser ("cuda" ou "cpu").
        save_path (str): Chemin pour sauvegarder le modèle.
    """
    # Optimiseur et scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_epochs * len(train_df) // batch_size,
    )
    
    # Boucle d'entraînement
    model.train()
    for epoch in range(num_epochs):
        print(f"=== Époque {epoch + 1}/{num_epochs} ===")
        progress_bar = tqdm(range(0, len(train_df), batch_size), desc="Entraînement")
        
        for i in progress_bar:
            # Génération d'un batch
            src_texts, tgt_texts = get_batch_pairs(train_df, batch_size)
            
            # Tokenisation des entrées et sorties
            inputs = tokenizer(src_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
            labels = tokenizer(tgt_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).input_ids
            
            # Déplacement sur le device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            
            # Passage avant
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            
            # Rétropropagation
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Mise à jour de la barre de progression
            progress_bar.set_postfix({"loss": loss.item()})
        
        # Sauvegarde du modèle après chaque époch
        model.save_pretrained(f"{save_path}/epoch_{epoch + 1}")
        tokenizer.save_pretrained(f"{save_path}/epoch_{epoch + 1}")