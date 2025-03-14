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
from moore_tsr.eval import create_loss_visualization

def cleanup():
    """Clear memory to prevent CUDA out-of-memory errors."""
    gc.collect()
    torch.cuda.empty_cache()

def tokenize_batch(tokenizer, texts, lang_code, max_length, device):
    """Tokenize a batch of texts with appropriate language code."""
    inputs = tokenizer(
        texts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=max_length,
        src_lang=lang_code if lang_code else None
    )
    return {k: v.to(device) for k, v in inputs.items()}

def get_batch_pairs(data, batch_size):
    LANGS = [ ('french', "fr_Latn"), ('moore', "moor_Latn")]
    (l1, long1), (l2, long2) = random.sample(LANGS, 2)
    xx, yy = [], []
    for _ in range(batch_size):
        item = data.iloc[random.randint(0, len(data)-1)]
        xx.append(item[l1])
        yy.append(item[l2])
    return xx, yy, long1, long2

def evaluate_model(model, tokenizer, eval_df, batch_size=16, max_length=128, device="cuda"):
    """Evaluate model on validation data and return loss."""
    model.eval()
    total_loss = 0.0
    batches = 0
    
    with torch.no_grad():
        for i in range(0, len(eval_df), batch_size):
            try:
                src_texts, tgt_texts, src_lang, tgt_lang = get_batch_pairs(
                    eval_df, batch_size
                )
                print(src_texts)
                with autocast():
                    tokenizer.src_lang = src_lang
                    x = tokenize_batch(tokenizer, src_texts, src_lang, max_length, device)
                    tokenizer.tgt_lang = tgt_lang
                    y = tokenize_batch(tokenizer, tgt_texts, tgt_lang, max_length, device)
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

def train_model(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    num_epochs: int = 3,
    batch_size: int = 16,
    max_length: int = 128,
    learning_rate: float = 5e-5,
    warmup_steps: int = 1000,
    device: str = "cuda",
    save_path: str = "./models/nllb-moore-finetuned",
    accumulation_steps: int = 4,
    eval_steps: int = 1000,
    save_steps: int = 5000,
    early_stopping_patience: int = 5,
    max_grad_norm: float = 1.0,
    fp16: bool = True,
    resume_from: str = None,
) -> None:
    """
    Optimized training function for sequence-to-sequence models combining best practices.
    """
    # Create save directories
    os.makedirs(save_path, exist_ok=True)
    best_model_path = f"{save_path}_best"
    os.makedirs(best_model_path, exist_ok=True)
    
    # Check if Google Drive is mounted
    drive_mounted = os.path.exists("/content/drive/MyDrive")
    if drive_mounted:
        drive_save_path = f"/content/drive/MyDrive/{save_path}"
        drive_best_path = f"{drive_save_path}_best"
        os.makedirs(drive_save_path, exist_ok=True)
        os.makedirs(drive_best_path, exist_ok=True)
        logger.info(f"Google Drive is mounted. Models will also be saved to: {drive_save_path}")
    
    # Resume from checkpoint if specified
    if resume_from:
        logger.info(f"Resuming training from: {resume_from}")
        model = AutoModelForSeq2SeqLM.from_pretrained(resume_from).to(device)
        tokenizer = AutoTokenizer.from_pretrained(resume_from)
    
    # Initialize mixed precision training if enabled
    scaler = GradScaler() if fp16 else None
    
    # Calculate total steps for scheduler
    total_steps = (len(train_df) // batch_size // accumulation_steps) * num_epochs
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=float(learning_rate))
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    # Initialize tracking variables
    losses = []
    train_losses = []
    val_losses = []
    step_markers = []
    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0
    running_loss = 0.0
    
    # Initial evaluation
    logger.info("Performing initial evaluation...")
    model.train()
    cleanup()

    initial_val_loss = evaluate_model(
        model, tokenizer, val_df, batch_size, max_length, device
    )
    logger.info(f"Initial validation loss: {initial_val_loss:.4f}")
    best_val_loss = initial_val_loss
    val_losses.append(initial_val_loss)
    step_markers.append(0)
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        logger.info(f"=== Epoch {epoch + 1}/{num_epochs} ===")
        # Shuffle training data at the beginning of each epoch
        train_df = shuffle(train_df)
        
        progress_bar = tqdm(range(0, len(train_df), batch_size), desc=f"Epoch {epoch+1}")
        
        for i in progress_bar:
            # Zero gradients only at the beginning of accumulation cycle
            if global_step % accumulation_steps == 0:
                optimizer.zero_grad(set_to_none=True)
            
            try:
                # Get batch data
                src_texts, tgt_texts, src_langs, tgt_langs = get_batch_pairs(
                    train_df, batch_size
                )
                
                # Use mixed precision if enabled
                with autocast(enabled=fp16):
                    # Tokenize inputs and outputs
                    x = tokenize_batch(tokenizer, src_texts, src_langs, max_length, device)
                    y = tokenize_batch(tokenizer, tgt_texts, tgt_langs, max_length, device)
                    # Replace padding tokens with -100 to ignore in loss computation
                    y_input_ids[y_input_ids == tokenizer.pad_token_id] = -100
                    
                    # Forward pass
                    outputs = model(**x, labels=y_input_ids)
                    loss = outputs.loss / accumulation_steps
                
                # Backward pass with scaling if fp16 is enabled
                if fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Add to running loss for tracking
                running_loss += loss.item() * accumulation_steps
                
                # Update weights if accumulation cycle is complete
                if (global_step + 1) % accumulation_steps == 0:
                    if fp16:
                        # Unscale gradients for clipping
                        scaler.unscale_(optimizer)
                    
                    # Clip gradients to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                    
                    if fp16:
                        # Optimizer and scheduler steps with scaling if fp16 is enabled
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    
                    scheduler.step()
                    
                    # Record loss
                    losses.append(running_loss)
                    train_losses.append(running_loss)
                    running_loss = 0.0
                    
                    # Update progress bar
                    avg_loss = np.mean(losses[-100:]) if losses else 0
                    progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
                
                # Perform evaluation at regular intervals
                if global_step > 0 and global_step % eval_steps == 0:
                    val_loss = evaluate_model(
                        model, tokenizer, val_df, batch_size, max_length, device
                    )
                    val_losses.append(val_loss)
                    step_markers.append(global_step)
                    
                    logger.info(f"\nStep {global_step}, Validation Loss: {val_loss:.4f}")
                    
                    # Check for improvement and early stopping
                    if val_loss < best_val_loss:
                        logger.info(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}")
                        best_val_loss = val_loss
                        patience_counter = 0
                        
                        # Save best model
                        logger.info(f"Saving best model to {best_model_path}")
                        model.save_pretrained(best_model_path)
                        tokenizer.save_pretrained(best_model_path)
                        
                        # Save to Google Drive if mounted
                        if drive_mounted:
                            model.save_pretrained(drive_best_path)
                            tokenizer.save_pretrained(drive_best_path)
                    else:
                        patience_counter += 1
                        logger.info(f"Validation loss did not improve. Patience: {patience_counter}/{early_stopping_patience}")
                        
                        if patience_counter >= early_stopping_patience:
                            logger.info(f"Early stopping triggered after {global_step} steps!")
                            
                            # Create visualization and save final model before stopping
                            create_loss_visualization(
                                train_losses, val_losses, step_markers, 
                                save_path, drive_save_path if drive_mounted else None
                            )
                            
                            # Save final model
                            model.save_pretrained(save_path)
                            tokenizer.save_pretrained(save_path)
                            
                            # Save to Google Drive if mounted
                            if drive_mounted:
                                model.save_pretrained(drive_save_path)
                                tokenizer.save_pretrained(drive_save_path)
                            
                            return
                
                # Save model at regular intervals
                if global_step > 0 and global_step % save_steps == 0:
                    checkpoint_path = f"{save_path}/step_{global_step}"
                    os.makedirs(checkpoint_path, exist_ok=True)
                    
                    logger.info(f"\nSaving checkpoint to {checkpoint_path}")
                    model.save_pretrained(checkpoint_path)
                    tokenizer.save_pretrained(checkpoint_path)
                    
                    # Save to Google Drive if mounted
                    if drive_mounted:
                        drive_checkpoint_path = f"{drive_save_path}/step_{global_step}"
                        os.makedirs(drive_checkpoint_path, exist_ok=True)
                        model.save_pretrained(drive_checkpoint_path)
                        tokenizer.save_pretrained(drive_checkpoint_path)
                
                # Periodic cleanup to prevent memory issues
                if global_step % 100 == 0:
                    cleanup()
                
                global_step += 1
                
            except RuntimeError as e:
                logger.error(f"Error during training: {e}")
                cleanup()
                continue
        
        # End of epoch, save model
        epoch_save_path = f"{save_path}/epoch_{epoch+1}"
        os.makedirs(epoch_save_path, exist_ok=True)
        
        logger.info(f"\nSaving model after epoch {epoch+1} to {epoch_save_path}")
        model.save_pretrained(epoch_save_path)
        tokenizer.save_pretrained(epoch_save_path)
        
        # Save to Google Drive if mounted
        if drive_mounted:
            drive_epoch_save_path = f"{drive_save_path}/epoch_{epoch+1}"
            os.makedirs(drive_epoch_save_path, exist_ok=True)
            model.save_pretrained(drive_epoch_save_path)
            tokenizer.save_pretrained(drive_epoch_save_path)
    
    # Create visualization after training
    create_loss_visualization(
        train_losses, val_losses, step_markers, 
        save_path, drive_save_path if drive_mounted else None
    )
    
    # Save final model
    logger.info(f"\nSaving final model to {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    # Save to Google Drive if mounted
    if drive_mounted:
        model.save_pretrained(drive_save_path)
        tokenizer.save_pretrained(drive_save_path)

