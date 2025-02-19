"""
Script to analyze data transformations.
"""

import pandas as pd
from tabulate import tabulate
from transformers import AutoTokenizer
import argparse

from src.data.data_loader import load_data


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="""
        Analyze dataset transformations. This script examines:
        - Original training samples
        - Augmented samples (if any)
        - Validation samples
        - Test samples

        For each sample set, it provides:
        - Token counts and statistics
        - Text samples with their tokenization
        - Label distribution
        """
    )
    parser.add_argument(
        "--train_path",
        type=str,
        required=True,
        help="Path to the training dataset CSV file",
    )
    parser.add_argument(
        "--test_path", type=str, required=True, help="Path to the test dataset CSV file"
    )
    return parser.parse_args()


def analyze_dataset(
    train_path: str, test_path: str, model_name: str = "answerdotai/ModernBERT-base"
):
    """
    Analyze dataset transformations.

    Args:
        train_path: Path to training dataset
        test_path: Path to test dataset
        model_name: Name of the model (for tokenizer)
    """
    print("Loading data and applying transformations...")
    train_df, val_df, test_df = load_data(
        train_path=train_path,
        test_path=test_path,
        val_split=0.15,
        random_seed=42,
        augment=False,
    )

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Analyze original training samples
    print("\n=== Original Training Samples Analysis ===")
    original_samples = train_df.iloc[:10]  # First 10 samples
    analyze_samples(original_samples, tokenizer)

    # Find augmented samples
    augmented_df = train_df[~train_df.index.isin(original_samples.index)]
    print("\n=== Augmented Samples Analysis ===")
    augmented_samples = augmented_df.iloc[:10]  # First 10 augmented samples
    analyze_samples(augmented_samples, tokenizer)

    # Analyze validation samples
    print("\n=== Validation Samples Analysis ===")
    val_samples = val_df.iloc[:10]
    analyze_samples(val_samples, tokenizer)

    # Analyze test samples
    print("\n=== Test Samples Analysis ===")
    test_samples = test_df.iloc[:10]
    analyze_samples(test_samples, tokenizer)


def analyze_samples(df: pd.DataFrame, tokenizer):
    """Analyze and display sample transformations."""
    rows = []
    for _, row in df.iterrows():
        text = row["text"]
        label = row["sensitive_label"]

        # Get tokenization
        tokens = tokenizer.tokenize(text)
        # Clean up token display (remove Ġ and other special chars)
        clean_tokens = [t.replace("Ġ", " ").strip() for t in tokens]
        token_str = " ".join(clean_tokens)

        # Create readable token breakdown
        token_breakdown = " ".join(clean_tokens)

        rows.append(
            [
                "Sensitive" if label == 1 else "Not Sensitive",
                text[:100] + "..." if len(text) > 100 else text,
                len(tokens),
                token_breakdown[:100] + "..."
                if len(token_breakdown) > 100
                else token_breakdown,
                token_str[:100] + "..." if len(token_str) > 100 else token_str,
            ]
        )

    # Print analysis
    headers = [
        "Label",
        "Original Text",
        "Token Count",
        "Token Breakdown",
        "Reconstructed Text",
    ]
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    # Print tokenization statistics
    print("\nTokenization Statistics:")
    token_lengths = [len(tokenizer.tokenize(row["text"])) for _, row in df.iterrows()]
    stats = {
        "Average Token Length": sum(token_lengths) / len(token_lengths),
        "Min Token Length": min(token_lengths),
        "Max Token Length": max(token_lengths),
    }
    print(
        tabulate(
            [[k, f"{v:.2f}"] for k, v in stats.items()],
            headers=["Metric", "Value"],
            tablefmt="simple",
        )
    )


if __name__ == "__main__":
    args = parse_args()
    analyze_dataset(train_path=args.train_path, test_path=args.test_path)
