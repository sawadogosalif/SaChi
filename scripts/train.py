import argparse
import yaml
from moore_tsr.training.train import train_model
from moore_tsr.training.model_setup import setup_model_and_tokenizer
from moore_tsr.data.dataset import load_split_data
from loguru import logger
import os
import torch

def main():
    parser = argparse.ArgumentParser(description="Train the NLLB model for French-Moore translation.")
    parser.add_argument("--config", type=str, default="./configs/training.yaml", 
                        help="Path to the configuration file.")
    parser.add_argument("--model_name", type=str, 
                        help="Name of the pre-trained model to fine-tune (overrides config).")
    parser.add_argument("--batch_size", type=int, 
                        help="Batch size for training (overrides config).")
    parser.add_argument("--num_epochs", type=int, 
                        help="Number of training epochs (overrides config).")
    parser.add_argument("--learning_rate", type=float, 
                        help="Learning rate for the optimizer (overrides config).")
    parser.add_argument("--save_path", type=str, 
                        help="Path to save the fine-tuned model (overrides config).")
    parser.add_argument("--max_length", type=int, 
                        help="Maximum sequence length (overrides config).")
    parser.add_argument("--accumulation_steps", type=int, 
                        help="Gradient accumulation steps (overrides config).")
    parser.add_argument("--warmup_steps", type=int, 
                        help="Warmup steps (overrides config).")
    parser.add_argument("--eval_steps", type=int, 
                        help="Steps between evaluations (overrides config).")
    parser.add_argument("--save_steps", type=int, 
                        help="Steps between model saves (overrides config).")
    parser.add_argument("--early_stopping_patience", type=int, 
                        help="Patience for early stopping (overrides config).")
    parser.add_argument("--fp16", action="store_true", 
                        help="Use mixed precision training (overrides config).")
    parser.add_argument("--resume_from", type=str, 
                        help="Resume training from a checkpoint (overrides config).")
    args = parser.parse_args()
    
    # Load configuration from yaml file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments if provided
    model_name = args.model_name or config['model']['name']
    save_path = args.save_path or config['model']['save_path']
    
    # Training parameters
    batch_size = args.batch_size or config['training']['batch_size']
    num_epochs = args.num_epochs or config['training']['num_epochs']
    learning_rate = args.learning_rate or config['training']['learning_rate']
    max_length = args.max_length or config['training'].get('max_length', 128)
    warmup_steps = args.warmup_steps or config['training'].get('warmup_steps', 1000)
    
    # Advanced training parameters
    accumulation_steps = args.accumulation_steps or config['training'].get('accumulation_steps', 4)
    eval_steps = args.eval_steps or config['training'].get('eval_steps', 1000)
    save_steps = args.save_steps or config['training'].get('save_steps', 5000)
    early_stopping_patience = args.early_stopping_patience or config['training'].get('early_stopping_patience', 5)
    fp16 = args.fp16 or config['training'].get('fp16', True)
    resume_from = args.resume_from or config['training'].get('resume_from', None)
    
    # Log configuration
    logger.info(f"Model: {model_name}")
    logger.info(f"Save path: {save_path}")
    logger.info(f"Training parameters: batch_size={batch_size}, num_epochs={num_epochs}, "
                f"learning_rate={learning_rate}, max_length={max_length}")
    logger.info(f"Advanced training parameters: accumulation_steps={accumulation_steps}, "
                f"warmup_steps={warmup_steps}, eval_steps={eval_steps}, save_steps={save_steps}, "
                f"early_stopping_patience={early_stopping_patience}, fp16={fp16}")
    if resume_from:
        logger.info(f"Resuming from: {resume_from}")
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load data
    train_df, val_df, test_df = load_split_data()
    logger.info(f"Data loaded: {len(train_df)} training, {len(val_df)} validation, {len(test_df)} test examples")
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_name=model_name)
    
    # Train the model
    train_model(
        model=model,
        tokenizer=tokenizer,
        train_df=train_df,
        val_df=val_df,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_length=max_length,
        device=device,
        save_path=save_path,
        accumulation_steps=accumulation_steps,
        warmup_steps=warmup_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
        early_stopping_patience=early_stopping_patience,
        fp16=fp16,
        resume_from=resume_from,
    )

if __name__ == "__main__":
    main()
