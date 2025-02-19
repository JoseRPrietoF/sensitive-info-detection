"""
Training utilities with MLflow integration for experiment tracking.
"""

from typing import Dict, Tuple, List
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm
import mlflow
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    accuracy_score,
)
from tabulate import tabulate


class Trainer:
    """
    Trainer class for sensitive information detection model.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        experiment_name: str = "sensitive-info-detection",
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            optimizer: Optimizer for training
            device: Device to train on
            experiment_name: Name for MLflow experiment
        """
        super().__init__()
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.device = device

        # Compute and set class weights
        class_weights = model.compute_class_weights(train_loader)
        model.set_class_weights(class_weights)

        # Store weights for logging
        self.class_weights = class_weights

        # Set up MLflow
        # Use relative path
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment(experiment_name)

        # Store best metrics
        self.best_metrics = {"train": None, "val": None, "test": None, "epoch": 0}

    def analyze_predictions(
        self, data_loader: torch.utils.data.DataLoader, num_examples: int = 10
    ) -> List[Dict]:
        """
        Analyze model predictions on a dataset.

        Args:
            data_loader: DataLoader to analyze
            num_examples: Number of examples to show

        Returns:
            List of dictionaries containing prediction analysis
        """
        self.model.eval()
        all_examples = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(input_ids, attention_mask, labels)
                probabilities = torch.softmax(outputs.logits, dim=1)
                predictions = torch.argmax(probabilities, dim=1)

                # Convert tokens back to text
                texts = [
                    self.train_loader.dataset.tokenizer.decode(
                        ids, skip_special_tokens=True
                    )
                    for ids in input_ids
                ]

                # Store examples
                for text, pred, label, prob in zip(
                    texts, predictions, labels, probabilities
                ):
                    confidence = prob[pred].item()
                    all_examples.append(
                        {
                            "text": text,
                            "predicted": pred.item(),
                            "actual": label.item(),
                            "confidence": confidence,
                            "correct": pred.item() == label.item(),
                        }
                    )

        # Sort by confidence and get interesting examples
        correct_predictions = [ex for ex in all_examples if ex["correct"]]
        incorrect_predictions = [ex for ex in all_examples if not ex["correct"]]

        # Get a mix of correct and incorrect predictions
        num_each = num_examples // 2
        selected_examples = (
            sorted(incorrect_predictions, key=lambda x: x["confidence"], reverse=True)[
                :num_each
            ]
            + sorted(correct_predictions, key=lambda x: x["confidence"], reverse=True)[
                :num_each
            ]
        )

        return selected_examples

    def _print_prediction_analysis(self, examples: List[Dict], save_dir: Path) -> None:
        """Print prediction analysis in a formatted table and save to file."""
        headers = ["Prediction", "Actual", "Confidence", "Text"]
        rows = []

        for ex in examples:
            # Don't truncate text for file output
            rows.append(
                [
                    "Sensitive" if ex["predicted"] == 1 else "Not Sensitive",
                    "Sensitive" if ex["actual"] == 1 else "Not Sensitive",
                    f"{ex['confidence']:.4f}",
                    ex["text"],
                ]
            )

        # Print to console with truncated text
        console_rows = []
        for row in rows:
            text = row[3]
            if len(text) > 100:
                text = text[:97] + "..."
            console_rows.append([row[0], row[1], row[2], text])

        print("\nPrediction Analysis:")
        print(tabulate(console_rows, headers=headers, tablefmt="grid"))

        # Print summary statistics
        correct = sum(1 for ex in examples if ex["correct"])
        total = len(examples)
        summary_stats = "\nAnalysis Summary:\n"
        summary_stats += f"Correct: {correct}/{total} ({correct/total*100:.1f}%)\n"
        summary_stats += (
            f"Average Confidence: {sum(ex['confidence'] for ex in examples)/total:.4f}"
        )

        print(summary_stats)

        # Save to file
        analysis_file = save_dir / "prediction_analysis.txt"
        with open(analysis_file, "w") as f:
            f.write("Prediction Analysis\n")
            f.write("===================\n\n")
            f.write(tabulate(rows, headers=headers, tablefmt="grid"))
            f.write("\n\n")
            f.write(summary_stats)

            # Add detailed error analysis
            f.write("\n\nDetailed Error Analysis\n")
            f.write("=====================\n\n")

            # Analyze false positives
            false_positives = [
                ex for ex in examples if ex["predicted"] == 1 and ex["actual"] == 0
            ]
            f.write(
                f"\nFalse Positives (Incorrectly labeled as Sensitive): {len(false_positives)}\n"
            )
            if false_positives:
                for i, ex in enumerate(false_positives, 1):
                    f.write(f"\n{i}. Confidence: {ex['confidence']:.4f}\n")
                    f.write(f"   Text: {ex['text']}\n")

            # Analyze false negatives
            false_negatives = [
                ex for ex in examples if ex["predicted"] == 0 and ex["actual"] == 1
            ]
            f.write(
                f"\nFalse Negatives (Missed Sensitive Information): {len(false_negatives)}\n"
            )
            if false_negatives:
                for i, ex in enumerate(false_negatives, 1):
                    f.write(f"\n{i}. Confidence: {ex['confidence']:.4f}\n")
                    f.write(f"   Text: {ex['text']}\n")

            # Add confidence distribution
            f.write("\n\nConfidence Distribution\n")
            f.write("======================\n")
            f.write("\nCorrect Predictions:\n")
            correct_confidences = [ex["confidence"] for ex in examples if ex["correct"]]
            if correct_confidences:
                f.write(
                    f"Average: {sum(correct_confidences)/len(correct_confidences):.4f}\n"
                )
                f.write(f"Min: {min(correct_confidences):.4f}\n")
                f.write(f"Max: {max(correct_confidences):.4f}\n")

            f.write("\nIncorrect Predictions:\n")
            incorrect_confidences = [
                ex["confidence"] for ex in examples if not ex["correct"]
            ]
            if incorrect_confidences:
                f.write(
                    f"Average: {sum(incorrect_confidences)/len(incorrect_confidences):.4f}\n"
                )
                f.write(f"Min: {min(incorrect_confidences):.4f}\n")
                f.write(f"Max: {max(incorrect_confidences):.4f}\n")

    def train_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Train the model for one epoch with weighted loss."""
        self.model.train()
        total_loss = 0
        predictions = []
        probabilities = []
        targets = []

        for batch in tqdm(self.train_loader, desc="Training"):
            self.optimizer.zero_grad()

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)

            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            # Calculate probabilities
            probs = torch.softmax(outputs.logits, dim=1)

            # Apply class weights to loss using the tensor version
            if hasattr(self.model, "class_weights_tensor"):
                weights = self.model.class_weights_tensor
                loss = outputs.loss * weights[labels].to(self.device)
                loss = loss.mean()
            else:
                loss = outputs.loss

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            predictions.extend(outputs.logits.argmax(dim=1).cpu().numpy())
            probabilities.extend(
                probs[:, 1].cpu().detach().numpy()
            )  # Probability of positive class
            targets.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(self.train_loader)
        metrics = self._compute_metrics(predictions, probabilities, targets)

        return avg_loss, metrics

    def evaluate(
        self, data_loader: torch.utils.data.DataLoader, desc: str = "Evaluation"
    ) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate the model on a dataset.

        Args:
            data_loader: DataLoader for evaluation
            desc: Description for progress bar

        Returns:
            Tuple of (average loss, metrics dictionary)
        """
        self.model.eval()
        total_loss = 0
        predictions = []
        probabilities = []
        targets = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc=desc):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(input_ids, attention_mask, labels)
                probs = torch.softmax(outputs.logits, dim=1)

                loss = outputs.loss
                total_loss += loss.item()
                predictions.extend(outputs.logits.argmax(dim=1).cpu().numpy())
                probabilities.extend(
                    probs[:, 1].cpu().detach().numpy()
                )  # Probability of positive class
                targets.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(data_loader)
        metrics = self._compute_metrics(predictions, probabilities, targets)

        return avg_loss, metrics

    def validate(self) -> Tuple[float, Dict[str, float]]:
        """
        Validate the model.

        Returns:
            Tuple of (average loss, metrics dictionary)
        """
        return self.evaluate(self.val_loader, desc="Validation")

    def test(self) -> Tuple[float, Dict[str, float]]:
        """
        Test the model.

        Returns:
            Tuple of (average loss, metrics dictionary)
        """
        return self.evaluate(self.test_loader, desc="Testing")

    def _compute_metrics(
        self, predictions: list, probabilities: list, targets: list
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics.

        Args:
            predictions: List of model predictions (binary classifications)
            probabilities: List of probabilities for the positive class
            targets: List of ground truth labels (binary classifications)

        Returns:
            Dict[str, float]: Dictionary containing metrics
        """
        accuracy = accuracy_score(targets, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, predictions, average="binary", zero_division=0
        )

        try:
            auc_roc = roc_auc_score(targets, probabilities)
        except ValueError:
            auc_roc = 0.0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc_roc": auc_roc,
        }

    def _print_metrics(
        self,
        epoch: int,
        train_loss: float,
        train_metrics: Dict[str, float],
        val_loss: float,
        val_metrics: Dict[str, float],
    ) -> None:
        """
        Print training and validation metrics in a formatted table.

        Args:
            epoch (int): Current epoch number
            train_loss (float): Average training loss
            train_metrics (Dict[str, float]): Dictionary of training metrics
            val_loss (float): Average validation loss
            val_metrics (Dict[str, float]): Dictionary of validation metrics
        """
        headers = ["Metric", "Training", "Validation"]
        metrics_data = [
            ["Loss", f"{train_loss:.4f}", f"{val_loss:.4f}"],
            [
                "Accuracy",
                f"{train_metrics['accuracy']:.4f}",
                f"{val_metrics['accuracy']:.4f}",
            ],
            [
                "Precision",
                f"{train_metrics['precision']:.4f}",
                f"{val_metrics['precision']:.4f}",
            ],
            [
                "Recall",
                f"{train_metrics['recall']:.4f}",
                f"{val_metrics['recall']:.4f}",
            ],
            ["F1-Score", f"{train_metrics['f1']:.4f}", f"{val_metrics['f1']:.4f}"],
            [
                "ROC-AUC",
                f"{train_metrics['auc_roc']:.4f}",
                f"{val_metrics['auc_roc']:.4f}",
            ],
        ]

        print(f"\nEpoch {epoch+1} Metrics:")
        print(tabulate(metrics_data, headers=headers, tablefmt="grid"))

    def _print_final_summary(self) -> None:
        """
        Print final summary of best metrics achieved during training.
        Displays a formatted table comparing training, validation, and test metrics.
        """
        print("\n" + "=" * 50)
        print("Training Complete - Best Model Performance")
        print("=" * 50)

        headers = ["Metric", "Training", "Validation", "Test"]
        metrics_data = [
            [
                "Accuracy",
                f"{self.best_metrics['train']['accuracy']:.4f}",
                f"{self.best_metrics['val']['accuracy']:.4f}",
                f"{self.best_metrics['test']['accuracy']:.4f}",
            ],
            [
                "Precision",
                f"{self.best_metrics['train']['precision']:.4f}",
                f"{self.best_metrics['val']['precision']:.4f}",
                f"{self.best_metrics['test']['precision']:.4f}",
            ],
            [
                "Recall",
                f"{self.best_metrics['train']['recall']:.4f}",
                f"{self.best_metrics['val']['recall']:.4f}",
                f"{self.best_metrics['test']['recall']:.4f}",
            ],
            [
                "F1-Score",
                f"{self.best_metrics['train']['f1']:.4f}",
                f"{self.best_metrics['val']['f1']:.4f}",
                f"{self.best_metrics['test']['f1']:.4f}",
            ],
            [
                "ROC-AUC",
                f"{self.best_metrics['train']['auc_roc']:.4f}",
                f"{self.best_metrics['val']['auc_roc']:.4f}",
                f"{self.best_metrics['test']['auc_roc']:.4f}",
            ],
        ]

        print(tabulate(metrics_data, headers=headers, tablefmt="grid"))
        print(f"\nBest model saved from epoch: {self.best_metrics['epoch']}")
        print("=" * 50)

    def train(
        self,
        num_epochs: int,
        save_dir: str = "models",
        early_stopping_patience: int = 3,
        tokenizer=None,
    ) -> None:
        """
        Train the model for specified number of epochs with early stopping.

        Args:
            num_epochs (int): Maximum number of epochs to train
            save_dir (str): Directory path to save model checkpoints and analysis
            early_stopping_patience (int): Number of epochs to wait for improvement before stopping
            tokenizer: The tokenizer used for the model

        Notes:
            - Uses MLflow for experiment tracking
            - Saves best model based on validation F1 score
            - Performs prediction analysis on test set
            - Logs metrics, model artifacts, and prediction analysis to MLflow
        """
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)

        best_val_f1 = 0
        patience_counter = 0
        best_model_path = None

        # End any existing runs before starting a new one
        mlflow.end_run()

        with mlflow.start_run():
            try:
                # Save tokenizer locally if provided
                if tokenizer is not None:
                    tokenizer.save_pretrained(save_path / "tokenizer")
                    mlflow.log_artifacts(str(save_path / "tokenizer"), "tokenizer")

                # Log model parameters
                mlflow.log_params(
                    {
                        "model_name": self.model.__class__.__name__,
                        "optimizer": self.optimizer.__class__.__name__,
                        "num_epochs": num_epochs,
                        "batch_size": self.train_loader.batch_size,
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                    }
                )

                for epoch in range(num_epochs):
                    # Training
                    train_loss, train_metrics = self.train_epoch()

                    # Validation
                    val_loss, val_metrics = self.validate()

                    # Print metrics
                    self._print_metrics(
                        epoch, train_loss, train_metrics, val_loss, val_metrics
                    )

                    # Log metrics for this epoch
                    mlflow.log_metrics(
                        {
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "train_accuracy": train_metrics["accuracy"],
                            "train_precision": train_metrics["precision"],
                            "train_recall": train_metrics["recall"],
                            "train_f1": train_metrics["f1"],
                            "train_auc_roc": train_metrics["auc_roc"],
                            "val_accuracy": val_metrics["accuracy"],
                            "val_precision": val_metrics["precision"],
                            "val_recall": val_metrics["recall"],
                            "val_f1": val_metrics["f1"],
                            "val_auc_roc": val_metrics["auc_roc"],
                        },
                        step=epoch,
                    )

                    # Early stopping and model saving
                    if val_metrics["f1"] > best_val_f1:
                        best_val_f1 = val_metrics["f1"]
                        patience_counter = 0

                        # Save model locally
                        best_model_path = save_path / "best_model.pt"
                        torch.save(self.model.state_dict(), best_model_path)

                        # Store best metrics
                        self.best_metrics["train"] = train_metrics
                        self.best_metrics["val"] = val_metrics
                        self.best_metrics["epoch"] = epoch + 1

                        # Log best metrics
                        mlflow.log_metrics(
                            {
                                "best_epoch": epoch + 1,
                                "best_val_f1": val_metrics["f1"],
                                "best_val_accuracy": val_metrics["accuracy"],
                                "best_val_precision": val_metrics["precision"],
                                "best_val_recall": val_metrics["recall"],
                                "best_val_auc_roc": val_metrics["auc_roc"],
                            }
                        )
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            print(
                                f"\nEarly stopping triggered after {epoch + 1} epochs!"
                            )
                            break

                # Evaluate on test set
                print("\nEvaluating best model on test set...")
                self.model.load_state_dict(torch.load(best_model_path))
                test_loss, test_metrics = self.test()
                self.best_metrics["test"] = test_metrics

                # Log final test metrics
                mlflow.log_metrics(
                    {
                        "test_loss": test_loss,
                        "test_accuracy": test_metrics["accuracy"],
                        "test_precision": test_metrics["precision"],
                        "test_recall": test_metrics["recall"],
                        "test_f1": test_metrics["f1"],
                        "test_auc_roc": test_metrics["auc_roc"],
                    }
                )

                # Print final summary
                self._print_final_summary()

                # Analyze predictions
                print("\nAnalyzing model predictions...")
                examples = self.analyze_predictions(self.test_loader)
                self._print_prediction_analysis(examples, save_path)

                # Log prediction analysis file
                mlflow.log_artifact(str(save_path / "prediction_analysis.txt"))

                # # Save model to MLflow
                transformers_model = {"model": self.model.model, "tokenizer": tokenizer}
                mlflow.transformers.log_model(
                    transformers_model,
                    "model",
                    registered_model_name="sensitive_info_detector",
                    task="text-classification",
                )

                # Log the run ID for easy reference
                run_id = mlflow.active_run().info.run_id
                print(f"\nMLflow Run ID: {run_id}")
                print("Save this run ID to use with the API!")

            finally:
                # Ensure the run is ended even if an exception occurs
                mlflow.end_run()
