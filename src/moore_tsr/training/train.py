import pandas as pd
import torch
from torch.optim import AdamW
from transformers import get_scheduler, AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm
from moore_tsr.data.dataset import get_batch_pairs
from loguru import logger
import gc
import plotly.express as px
import os
from sklearn.utils import shuffle

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
    resume_from_epoch: int = 0,
    early_stopping_patience: int = 3,  # Nombre d'époques pour early stopping
) -> None:
    """
    Entraîne le modèle de traduction avec validation.

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
        early_stopping_patience (int): Nombre d'époques sans amélioration avant arrêt prématuré.
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
    
    train_losses = []  
    val_losses = []   
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    
    model.train()
    for epoch in range(resume_from_epoch, num_epochs):
        print(f"=== Époque {epoch + 1}/{num_epochs} ===")
        train_df = shuffle(train_df)
        # Entraînement
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
            
            # Enregistrement de la perte brute
            train_losses.append(loss.item())
            progress_bar.set_postfix({"train_loss": loss.item()})
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i in range(0, len(val_df), batch_size):
                src_texts, tgt_texts = get_batch_pairs(val_df, batch_size)
                inputs = tokenizer(src_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
                labels = tokenizer(tgt_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).input_ids
                inputs = {k: v.to(device) for k, v in inputs.items()}
                labels = labels.to(device)
                outputs = model(**inputs, labels=labels)
                val_loss += outputs.loss.item()
        
        val_loss /= len(val_df) // batch_size
        val_losses.append(outputs.loss.item())
        
        print(f"training loss (moyenne) : {sum(train_losses[-len(train_df) // batch_size:]) / (len(train_df) // batch_size):.4f}, "
              f"Val loss : {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping après {epoch + 1} époques sans amélioration.")
                break
        
        # Sauvegarde du modèle après chaque époque
        epoch_save_path = f"{save_path}/epoch_{epoch + 1}"
        model.save_pretrained(epoch_save_path)
        tokenizer.save_pretrained(epoch_save_path)
        
        # Sauvegarde sur Google Drive si monté
        if drive_mounted:
            drive_epoch_save_path = f"{drive_save_path}/epoch_{epoch + 1}"
            model.save_pretrained(drive_epoch_save_path)
            tokenizer.save_pretrained(drive_epoch_save_path)
    
    df_losses = pd.DataFrame({
        "Step": range(len(train_losses)),
        "Train Loss": train_losses,
        "Validation Loss": val_losses,
    })
        
    fig = px.line(df_losses, x="Step", y=["Train Loss", "Validation Loss"], 
                  title="Évolution des pertes pendant l'entraînement",
                  labels={"value": "loss", "variable": "Légende"},
                  template="plotly_white")
    
    fig.show()
    workspace_plot_path = f"{save_path}/training_validation_loss.html"
    fig.write_html(workspace_plot_path)
    
    # Sauvegarde du graphique sur Google Drive si monté
    if drive_mounted:
        drive_plot_path = f"{drive_save_path}/training_validation_loss.html"
        fig.write_html(drive_plot_path)
