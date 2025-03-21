import os
import gc
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import get_scheduler, AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm, trange
import plotly.express as px
from sklearn.utils import shuffle
from loguru import logger
import random
from moore_tsr.eval import create_loss_visualization, evaluate_model_loss
from moore_tsr.utils.helpers import save_model_and_tokenizer, get_batch_pairs, cleanup

from transformers.optimization import Adafactor
from transformers import get_constant_schedule_with_warmup

def tokenize_batch(tokenizer, texts, lang_code, max_length, device):
    """Tokenize a batch of texts with appropriate language code."""
    tokenizer.src_lang = lang_code
    inputs = tokenizer(
        texts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=max_length,
    )
    return inputs.to(device)



def train_model(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    num_epochs: int = 3,
    batch_size: int = 16,
    max_length: int = 128,
    learning_rate: float = 1e-4,
    warmup_steps: int = 1000,
    device: str = "cuda",
    save_path: str = "./models/nllb-moore-finetuned",
    accumulation_steps: int = 1,
    eval_steps: int = 1000,
    save_steps: int = 5000,
    early_stopping_patience: int = 5,
    max_grad_norm: float = 1.0,
    fp16: bool = False,
    resume_from: str = None,
) -> None:
    """
    Uses Adafactor optimizer instead of AdamW for memory efficiency.
    """
    best_model_path = f"{save_path}_best"
    drive_save_path = f"/content/drive/MyDrive/{save_path}"
    drive_best_path = f"{drive_save_path}_best"
    logger.info(f"Google Drive is mounted. Models will also be saved to: {drive_save_path}")
    
    if resume_from:
        logger.info(f"Resuming training from: {resume_from}")
        model = AutoModelForSeq2SeqLM.from_pretrained(resume_from).to(device)
        tokenizer = AutoTokenizer.from_pretrained(resume_from)
    
    scaler = GradScaler() if fp16 else None
    total_steps = (len(train_df) // batch_size // accumulation_steps) * num_epochs
    
    optimizer = Adafactor(
        model.parameters(),
        lr=float(learning_rate),
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False
    )
    
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
    
    losses, train_losses, val_losses, step_markers = [], [], [], []
    best_val_loss, patience_counter, global_step, running_loss = float("inf"), 0, 0, 0.0
    
    logger.info("Performing initial evaluation...")
    model.train()
    cleanup()
    initial_val_loss = evaluate_model_loss(model, tokenizer, val_df, batch_size, max_length, device)
    logger.info(f"Initial validation loss: {initial_val_loss:.4f}")
    best_val_loss = initial_val_loss
    val_losses.append(initial_val_loss)
    step_markers.append(0)
    
    for epoch in range(num_epochs):
        logger.info(f"=== Epoch {epoch + 1}/{num_epochs} ===")
        train_df = shuffle(train_df)
        progress_bar = tqdm(range(0, len(train_df), batch_size), desc=f"Epoch {epoch+1}")
        
        for i in progress_bar:
            # Reset gradients at the beginning of accumulation 
            cleanup()
            if global_step % accumulation_steps == 0:
                optimizer.zero_grad(set_to_none=True)
            
            try:
                # Get batch data
                src_texts, tgt_texts, src_langs, tgt_langs = get_batch_pairs(train_df, batch_size)
                
                # Use mixed precision if enabled
                with autocast(enabled=fp16):
                    # Tokenize source and target texts
                    x = tokenize_batch(tokenizer, src_texts, src_langs, max_length, device)
                    y = tokenize_batch(tokenizer, tgt_texts, tgt_langs, max_length, device)
                    
                    # Replace padding tokens with -100 for loss calculation
                    y.input_ids[y.input_ids == tokenizer.pad_token_id] = -100
                    
                    # Forward pass
                    outputs = model(**x, labels=y.input_ids)
                    loss = outputs.loss / accumulation_steps
                
                # Backward pass with scaling if fp16 is enabled
                if fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Track running loss
                running_loss += loss.item() * accumulation_steps
                
                # Update parameters after accumulation steps
                if (global_step + 1) % accumulation_steps == 0:
                    # Unscale gradients for gradient clipping when using fp16
                    if fp16:
                        scaler.unscale_(optimizer)
                    
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                    
                    # Step optimizer and scaler
                    if fp16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    
                    # Step scheduler
                    scheduler.step()
                    
                    # Record loss
                    losses.append(running_loss)
                    train_losses.append(running_loss)
                    running_loss = 0.0
                
                # Evaluate periodically
                if global_step > 0 and global_step % eval_steps == 0:
                    val_loss = evaluate_model_loss(model, tokenizer, val_df, batch_size, max_length, device)
                    val_losses.append(val_loss)
                    step_markers.append(global_step)
                    
                    logger.info(f"Step {global_step}, Validation Loss: {val_loss:.4f}")
                    
                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        logger.info(f"Saving best model to {best_model_path}")
                        save_model_and_tokenizer(model, tokenizer, best_model_path, drive_best_path)
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            create_loss_visualization(train_losses, val_losses, step_markers, save_path, drive_save_path)
                            save_model_and_tokenizer(model, tokenizer, save_path, drive_save_path)
                            return
                
                # Save checkpoint periodically
                if global_step > 0 and global_step % save_steps == 0:
                    checkpoint_path = f"{save_path}/step_{global_step}"
                    logger.info(f"Saving checkpoint to {checkpoint_path}")
                    save_model_and_tokenizer(model, tokenizer, checkpoint_path, f"{drive_save_path}/step_{global_step}")
                
                # Periodic memory cleanup
                if global_step % 100 == 0:
                    cleanup()
                
                global_step += 1
                
            except RuntimeError as e:
                logger.error(f"Error during training: {e}")
                optimizer.zero_grad(set_to_none=True)
                cleanup()
                continue
        
    # Save final model and create visualization
    create_loss_visualization(train_losses, val_losses, step_markers, save_path, drive_save_path)
    save_model_and_tokenizer(model, tokenizer, save_path, drive_save_path)
