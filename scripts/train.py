import argparse
from src.training.train import train_model
from src.training.model_setup import setup_model_and_tokenizer
from src.data.dataset import load_split_data

def main():

    parser = argparse.ArgumentParser(description="Train the NLLB model for French-Moore translation.")
    parser.add_argument("--model_name", type=str, default="facebook/nllb-200-distilled-600M",
                        help="Name of the pre-trained model to fine-tune.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training.")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate for the optimizer.")
    parser.add_argument("--save_path", type=str, default="./models/nllb-moore-finetuned",
                        help="Path to save the fine-tuned model.")
    args = parser.parse_args()

    train_df, val_df, test_df = load_split_data()

    model, tokenizer = setup_model_and_tokenizer(model_name=args.model_name)

    train_model(
        model=model,
        tokenizer=tokenizer,
        train_df=train_df,
        val_df=val_df,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_path=args.save_path,
    )

if __name__ == "__main__":
    main()