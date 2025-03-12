# scripts/evaluate.py

import argparse
import os
import json
import yaml
from moore_tsr.eval.metrics import evaluate_model_with_bleu
from moore_tsr.training.model_setup import setup_model_and_tokenizer
from moore_tsr.data.dataset import load_split_data

def load_config(config_path: str) -> dict:
    """Load configuration from a YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned NLLB model.")
    parser.add_argument("--config", type=str, default="configs/training.yaml",
                        help="Path to the configuration file.")
    args = parser.parse_args()

    # Resolve the absolute path to the config file
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
    config_path = os.path.join(script_dir, "..", args.config)  # Resolve relative path
    config_path = os.path.abspath(config_path)  # Convert to absolute path

    # Load configuration
    config = load_config(config_path)
    model_path = config["model"]["save_path"]

    # Load the test dataset
    _, _, test_df = load_split_data(
        dataset_name=config["data"]["dataset_name"],
        train_size=config["data"]["train_size"],
        test_size=config["data"]["test_size"],
        val_size=config["data"]["val_size"],
        random_seed=config["data"]["random_seed"],
    )

    # Set up the model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(
        model_name=model_path,
    )

    # Evaluate the model
    bleu_results = evaluate_model_with_bleu(
        model=model,
        tokenizer=tokenizer,
        df_test=test_df,
    )
    print(f"BLEU Score (French → Moore): {bleu_results['fr_to_moore_bleu']:.2f}")
    print(f"BLEU Score (Moore → French): {bleu_results['moore_to_fr_bleu']:.2f}")
    print(f"Average BLEU Score: {bleu_results['average_bleu']:.2f}")
    with open(f"{model_path}/bleu_results.json", "w") as f:
        json.dump(bleu_results, f, indent=4)
if __name__ == "__main__":
    main()
