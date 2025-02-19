"""
Baseline models for sensitive information detection using sklearn.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tabulate import tabulate


def load_and_split_data(
    train_path: str, test_path: str, val_split: float = 0.15, random_seed: int = 42
):
    """Load and split data into train, validation, and test sets."""
    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Split training data into train and validation
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_split,
        stratify=train_df["sensitive_label"],
        random_state=random_seed,
    )

    print("\nDataset splits:")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")

    print("\nClass distribution:")
    print("\nTraining:")
    print(train_df["sensitive_label"].value_counts(normalize=True))
    print("\nValidation:")
    print(val_df["sensitive_label"].value_counts(normalize=True))
    print("\nTest:")
    print(test_df["sensitive_label"].value_counts(normalize=True))

    return train_df, val_df, test_df


def create_baseline_models():
    """Create dictionary of baseline models to evaluate."""
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42
        ),
        "Linear SVM": SVC(
            class_weight="balanced", max_iter=1000, random_state=42, probability=True
        ),
    }


def create_pipeline(model):
    """Create a pipeline with TF-IDF and classifier."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=10000, ngram_range=(1, 2), stop_words="english"
                ),
            ),
            ("classifier", model),
        ]
    )


def evaluate_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    model_name: str,
    save_dir: Path,
):
    """Train and evaluate a model, saving results."""
    # Train model
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)

    # Get predictions
    y_val_pred = model.predict(X_val)
    y_val_pred_proba = model.predict_proba(X_val)[:, 1]
    y_test_pred = model.predict(X_test)
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]

    # Get training predictions
    y_train_pred = model.predict(X_train)
    y_train_pred_proba = model.predict_proba(X_train)[:, 1]
    # Compute metrics
    train_metrics = compute_metrics(y_train, y_train_pred, y_train_pred_proba)
    val_metrics = compute_metrics(y_val, y_val_pred, y_val_pred_proba)
    test_metrics = compute_metrics(y_test, y_test_pred, y_test_pred_proba)

    # Print metrics in the same format as LLM experiments
    print("\n" + "=" * 50)
    print(f"{model_name} Performance")
    print("=" * 50)

    headers = ["Metric", "Training", "Validation", "Test"]
    metrics_data = [
        [
            "Accuracy",
            f"{train_metrics['accuracy']:.4f}",
            f"{val_metrics['accuracy']:.4f}",
            f"{test_metrics['accuracy']:.4f}",
        ],
        [
            "Precision",
            f"{train_metrics['precision']:.4f}",
            f"{val_metrics['precision']:.4f}",
            f"{test_metrics['precision']:.4f}",
        ],
        [
            "Recall",
            f"{train_metrics['recall']:.4f}",
            f"{val_metrics['recall']:.4f}",
            f"{test_metrics['recall']:.4f}",
        ],
        [
            "F1-Score",
            f"{train_metrics['f1']:.4f}",
            f"{val_metrics['f1']:.4f}",
            f"{test_metrics['f1']:.4f}",
        ],
        [
            "ROC-AUC",
            f"{train_metrics['auc_roc']:.4f}",
            f"{val_metrics['auc_roc']:.4f}",
            f"{test_metrics['auc_roc']:.4f}",
        ],
    ]

    print(tabulate(metrics_data, headers=headers, tablefmt="grid"))
    print("=" * 50)

    # Analyze test set errors
    print("\nTest Set Error Analysis:")
    print("-" * 30)

    # Convert test data to list for easier analysis
    X_test_list = X_test.tolist() if isinstance(X_test, pd.Series) else X_test

    # Find false positives (predicted sensitive when not)
    false_positives = [
        (text, pred, true)
        for text, pred, true in zip(X_test_list, y_test_pred, y_test)
        if pred == 1 and true == 0
    ]

    # Find false negatives (predicted not sensitive when it is)
    false_negatives = [
        (text, pred, true)
        for text, pred, true in zip(X_test_list, y_test_pred, y_test)
        if pred == 0 and true == 1
    ]

    print(f"\nTotal Test Examples: {len(y_test)}")
    print(f"Total Errors: {len(false_positives) + len(false_negatives)}")
    print(f"False Positives: {len(false_positives)}")
    print(f"False Negatives: {len(false_negatives)}")

    # Print example errors
    if false_positives:
        print("\nFalse Positive Examples (Incorrectly flagged as sensitive):")
        print("-" * 50)
        for text, _, _ in false_positives[:3]:  # Show up to 3 examples
            print(f"Text: {text[:200]}...")

    if false_negatives:
        print("\nFalse Negative Examples (Missed sensitive information):")
        print("-" * 50)
        for text, _, _ in false_negatives[:3]:  # Show up to 3 examples
            print(f"Text: {text[:200]}...")

    # Save error analysis to file
    error_file = save_dir / f"{model_name.lower()}_error_analysis.txt"
    with open(error_file, "w") as f:
        f.write(f"{model_name} Error Analysis\n")
        f.write("=" * 50 + "\n\n")

        f.write("Test Set Statistics:\n")
        f.write(f"Total Test Examples: {len(y_test)}\n")
        f.write(f"Total Errors: {len(false_positives) + len(false_negatives)}\n")
        f.write(f"False Positives: {len(false_positives)}\n")
        f.write(f"False Negatives: {len(false_negatives)}\n\n")

        if false_positives:
            f.write("\nFalse Positive Examples (Incorrectly flagged as sensitive):\n")
            f.write("-" * 50 + "\n")
            for text, _, _ in false_positives:
                f.write(f"Text: {text}\n\n")

        if false_negatives:
            f.write("\nFalse Negative Examples (Missed sensitive information):\n")
            f.write("-" * 50 + "\n")
            for text, _, _ in false_negatives:
                f.write(f"Text: {text}\n\n")

    # Save metrics to file
    metrics_file = save_dir / f"{model_name.lower()}_metrics.txt"
    with open(metrics_file, "w") as f:
        f.write(f"{model_name} Performance\n")
        f.write("=" * 50 + "\n")
        f.write(tabulate(metrics_data, headers=headers, tablefmt="grid"))
        f.write("\n" + "=" * 50 + "\n")

    # Plot and save confusion matrices
    plot_confusion_matrix(
        y_val,
        y_val_pred,
        f"{model_name} Validation",
        save_dir / f"{model_name.lower()}_val_cm.png",
    )
    plot_confusion_matrix(
        y_test,
        y_test_pred,
        f"{model_name} Test",
        save_dir / f"{model_name.lower()}_test_cm.png",
    )


def compute_metrics(y_true, y_pred, y_pred_proba):
    """Compute all metrics in the same format as LLM experiments."""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    try:
        auc_roc = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        auc_roc = 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_roc": auc_roc,
    }


def plot_confusion_matrix(y_true, y_pred, title: str, save_path: Path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    # Set up directories
    results_dir = Path("results/baseline")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_df, val_df, test_df = load_and_split_data(
        train_path="simulated_dataset/train_relabelled_anon.csv",
        test_path="simulated_dataset/validation_anon.csv",
        val_split=0.15,
        random_seed=42,
    )

    # Get text and labels
    X_train = train_df["text"]
    y_train = train_df["sensitive_label"]
    X_val = val_df["text"]
    y_val = val_df["sensitive_label"]
    X_test = test_df["text"]
    y_test = test_df["sensitive_label"]

    # Create and evaluate models
    models = create_baseline_models()
    for model_name, model in models.items():
        pipeline = create_pipeline(model)
        evaluate_model(
            pipeline,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            model_name,
            results_dir,
        )


if __name__ == "__main__":
    main()
