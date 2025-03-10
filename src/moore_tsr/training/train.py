import pandas as pd
import torch
from torch.optim import AdamW
from transformers import get_scheduler, AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm
from moore_tsr.data.dataset import get_batch_pairs
from loguru import logger
import gc
import matplotlib.pyplot as plt
import os

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
    save_path: str = "./models/finetuned",
    resume_from_epoch: int = 0
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
        resume_from_epoch (int): Époque à partir de laquelle reprendre l'entraînement.
    """
    # Check if Google Drive is mounted
    drive_mounted = os.path.exists("/content/drive/MyDrive")
    if drive_mounted:
        drive_save_path = f"/content/drive/MyDrive/{save_path}"
        os.makedirs(drive_save_path, exist_ok=True)
        print(f"Google Drive is mounted. Models will also be saved to: {drive_save_path}")
    
    # Charger le modèle et le tokenizer si on reprend l'entraînement
    if resume_from_epoch > 0:
        print(f"Reprise de l'entraînement à partir de l'époque {resume_from_epoch}...")
        model_path = f"{save_path}/epoch_{resume_from_epoch}"
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Optimiseur et scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_epochs * len(train_df) // batch_size,
    )
    
    losses = []
    
    # Boucle d'entraînement
    model.train()
    for epoch in range(resume_from_epoch, num_epochs):  # Commencer à l'époque spécifiée
        print(f"=== Époque {epoch + 1}/{num_epochs} ===")
        progress_bar = tqdm(range(0, len(train_df), batch_size), desc="Entraînement")
        
        for i in progress_bar:
            # Génération d'un batch
            torch.cuda.empty_cache()
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
            
            losses.append(loss.item())
            progress_bar.set_postfix({"loss": loss.item()})
        
        # Sauvegarde du modèle après chaque époque
        epoch_save_path = f"{save_path}/epoch_{epoch + 1}"
        model.save_pretrained(epoch_save_path)
        tokenizer.save_pretrained(epoch_save_path)
        
        # Sauvegarde sur Google Drive si monté
        if drive_mounted:
            drive_epoch_save_path = f"{drive_save_path}/epoch_{epoch + 1}"
            model.save_pretrained(drive_epoch_save_path)
            tokenizer.save_pretrained(drive_epoch_save_path)
    
    # Visualisation des pertes
    losses_series = pd.Series(losses)
    losses_ewm = losses_series.ewm(span=100).mean()  # Moyenne mobile exponentielle
    
    plt.figure(figsize=(12, 6))
    plt.plot(losses_series, label="Perte brute", alpha=0.5)
    plt.plot(losses_ewm, label="Moyenne mobile (span=100)", color="red")
    plt.title("Évolution de la perte pendant l'entraînement")
    plt.xlabel("Étapes")
    plt.ylabel("Perte")
    plt.legend()
    plt.grid(True)
    
    workspace_plot_path = f"{save_path}/training_loss.png"
    plt.savefig(workspace_plot_path)
    if drive_mounted:
        drive_plot_path = f"{drive_save_path}/training_loss.png"
        plt.savefig(drive_plot_path)
    
    plt.show()
