"""
Main training script for sensitive information detection model.
"""

import argparse
from pathlib import Path
import yaml
from datetime import datetime

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer

from src.data.data_loader import load_data, create_data_loaders, set_seed
from src.models.classifier import SensitiveInformationClassifier
from src.models.trainer import Trainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train sensitive information detection model"
    )

    # Data arguments
    parser.add_argument(
        "--train_path", type=str, required=True, help="Path to training dataset"
    )
    parser.add_argument(
        "--test_path", type=str, required=True, help="Path to test dataset"
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.15,
        help="Fraction of training data to use for validation",
    )
    parser.add_argument(
        "--random_seed", type=int, default=42, help="Random seed for reproducibility"
    )

    # Experiment arguments
    parser.add_argument(
        "--experiment_name", type=str, required=True, help="Name of the experiment"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="prajjwal1/bert-tiny",
        help="Name of the pre-trained model to use",
    )
    parser.add_argument(
        "--num_frozen_layers",
        type=int,
        default=4,
        help="Number of transformer layers to freeze (0 for none)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="Learning rate"
    )

    parser.add_argument(
        "--max_length", type=int, default=512, help="Maximum sequence length"
    )
    parser.add_argument(
        "--results_dir", type=str, default="results", help="Base directory for results"
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
        help="Number of epochs to wait before early stopping",
    )
    parser.add_argument(
        "--num_workers", type=int, default=2, help="Number of workers for data loading"
    )

    return parser.parse_args()


def save_config(args: argparse.Namespace, save_path: Path, tokenizer) -> None:
    """
    Save experiment configuration and tokenizer.

    Args:
        args: Parsed command line arguments
        save_path: Path to save configuration
        tokenizer: The tokenizer to save
    """
    config = {
        # Data configuration
        "train_path": str(args.train_path),
        "test_path": str(args.test_path),
        "val_split": args.val_split,
        "random_seed": args.random_seed,
        # Model configuration
        "experiment_name": args.experiment_name,
        "model_name": args.model_name,
        "num_frozen_layers": args.num_frozen_layers,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "early_stopping_patience": args.early_stopping_patience,
        "num_workers": args.num_workers,
        # Environment
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    }

    # Save config
    with open(save_path / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Save tokenizer
    tokenizer.save_pretrained(save_path / "tokenizer")


def main():
    """Main training function."""
    args = parse_args()

    # Set random seed
    set_seed(args.random_seed)

    # Set up experiment directory
    base_path = Path(args.results_dir)
    exp_path = base_path / args.experiment_name
    exp_path.mkdir(parents=True, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Add special tokens for sensitive information
    special_tokens = {
        "additional_special_tokens": [
            "[CARD_NUMBER]",
            "[DATE]",
            "[DOMAIN]",
            "[Name]",
            "[NUMBER]",
            "[PASSWORD]",
            "[SSN]",
            "[URL]",
        ]
    }
    # Add special tokens and resize embeddings
    num_added_tokens = tokenizer.add_special_tokens(special_tokens)
    print(f"Added {num_added_tokens} special tokens to the vocabulary")
    # Save configuration
    save_config(args, exp_path, tokenizer)

    # Load and preprocess data
    print("Loading data...")
    train_df, val_df, test_df = load_data(
        train_path=args.train_path,
        test_path=args.test_path,
        val_split=args.val_split,
        random_seed=args.random_seed,
        augment=False,
    )

    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        train_df,
        val_df,
        test_df,
        tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Initialize model
    print("Initializing model...")
    model = SensitiveInformationClassifier(
        model_name=args.model_name,
        num_classes=2,
        num_frozen_layers=args.num_frozen_layers,
    )
    # Resize token embeddings to match tokenizer
    model.model.resize_token_embeddings(len(tokenizer))

    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)

    # Initialize trainer with tokenizer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        device=device,
        experiment_name=args.experiment_name,
    )

    # Log class weights
    print("\nClass weights for handling imbalance:")
    print(f"Class 0 (Not Sensitive): {trainer.class_weights[0]:.4f}")
    print(f"Class 1 (Sensitive): {trainer.class_weights[1]:.4f}")

    # Train model with tokenizer
    print(f"\nStarting training experiment: {args.experiment_name}")
    print(f"Results will be saved in: {exp_path}")
    print("-" * 50)

    trainer.train(
        num_epochs=args.num_epochs,
        save_dir=str(exp_path),
        early_stopping_patience=args.early_stopping_patience,
        tokenizer=tokenizer,
    )

    print("\nTraining completed!")
    print(f"Results saved in: {exp_path}")


if __name__ == "__main__":
    main()
