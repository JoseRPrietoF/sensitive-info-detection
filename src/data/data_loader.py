"""
Data loading and preprocessing utilities for sensitive information detection.
"""

from typing import Tuple, Dict, Any

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import random


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SensitiveDataset(Dataset):
    """Custom Dataset for sensitive information detection."""

    def __init__(self, texts: list, labels: list, tokenizer: Any):
        """
        Initialize dataset.

        Args:
            texts: List of text samples
            labels: List of corresponding labels
            tokenizer: Tokenizer for text processing
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": label,
        }


def load_data(
    train_path: str,
    test_path: str,
    val_split: float = 0.15,
    random_seed: int = 42,
    augment: bool = False,  # TODO to be done
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load training and test datasets, and create validation split.

    Args:
        train_path: Path to training dataset
        test_path: Path to test dataset
        val_split: Fraction of training data to use for validation
        random_seed: Random seed for reproducibility
        augment: Whether to apply data augmentation

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Set random seed
    set_seed(random_seed)

    # Load datasets
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Create validation split
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_split,
        stratify=train_df["sensitive_label"],
        random_state=random_seed,
    )

    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print("\nDataset splits:")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")

    # Print class distribution
    print("\nClass distribution:")
    print("Training:")
    print(train_df["sensitive_label"].value_counts(normalize=True))
    print("\nValidation:")
    print(val_df["sensitive_label"].value_counts(normalize=True))
    print("\nTest:")
    print(test_df["sensitive_label"].value_counts(normalize=True))

    return train_df, val_df, test_df


def create_data_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer: Any,
    batch_size: int = 16,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training, validation and testing.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        tokenizer: Tokenizer for text processing
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for data loading

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = SensitiveDataset(
        texts=train_df["text"].tolist(),
        labels=train_df["sensitive_label"].tolist(),
        tokenizer=tokenizer,
    )

    val_dataset = SensitiveDataset(
        texts=val_df["text"].tolist(),
        labels=val_df["sensitive_label"].tolist(),
        tokenizer=tokenizer,
    )

    test_dataset = SensitiveDataset(
        texts=test_df["text"].tolist(),
        labels=test_df["sensitive_label"].tolist(),
        tokenizer=tokenizer,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
