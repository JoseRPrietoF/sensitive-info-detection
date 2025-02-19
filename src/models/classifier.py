"""
Model implementation for sensitive information detection.

This module provides a transformer-based classifier for detecting sensitive information
in text using pre-trained language models.
"""

from typing import Optional
from transformers import AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
import torch.nn as nn


class SensitiveInformationClassifier(nn.Module):
    """
    Transformer-based classifier for sensitive information detection.

    This classifier uses a pre-trained transformer model to detect sensitive information
    in text. It supports layer freezing for efficient fine-tuning and handles both
    training and inference scenarios.

    Attributes:
        model: The underlying transformer model for classification
    """

    def __init__(
        self,
        model_name: str = "prajjwal1/bert-tiny",
        num_classes: int = 2,
        num_frozen_layers: int = 4,
    ) -> None:
        """
        Initialize the classifier.

        Args:
            model_name (str): Name of the pre-trained model to use from HuggingFace hub
            num_classes (int): Number of output classes for classification
            num_frozen_layers (int): Number of initial transformer layers to freeze (0 for none)

        Returns:
            None
        """
        super().__init__()

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_classes
        )

        # Freeze layers if specified
        if num_frozen_layers > 0:
            self._freeze_layers(num_frozen_layers)

    def _freeze_layers(self, num_layers: int) -> None:
        """
        Freeze initial layers of the transformer for efficient fine-tuning.

        This method freezes the embedding layer and a specified number of encoder
        layers to prevent their weights from being updated during training.

        Args:
            num_layers (int): Number of encoder layers to freeze (0 for none)

        Returns:
            None
        """
        # Always freeze embeddings for stability
        for param in self.model.bert.embeddings.parameters():
            param.requires_grad = False

        # Freeze specified number of encoder layers
        if num_layers > 0:
            for layer in self.model.bert.encoder.layer[:num_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> SequenceClassifierOutput:
        """
        Forward pass of the model.

        Processes the input through the transformer model and returns classification
        outputs. When labels are provided, the loss is also computed and returned.

        Args:
            input_ids (torch.Tensor): Tensor of shape (batch_size, seq_length) containing input token IDs
            attention_mask (torch.Tensor): Tensor of shape (batch_size, seq_length) containing attention mask
            labels (Optional[torch.Tensor]): Optional tensor of shape (batch_size,) containing ground truth labels

        Returns:
            SequenceClassifierOutput: A dataclass containing:
                - loss (optional): Language modeling loss if labels provided
                - logits (torch.Tensor): Classification logits of shape (batch_size, num_classes)
                - hidden_states (optional): Model hidden states if output_hidden_states=True
                - attentions (optional): Model attentions if output_attentions=True
        """
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

        return outputs

    def compute_class_weights(self, train_loader):
        """Compute class weights based on training data distribution."""
        label_counts = torch.zeros(2)  # Binary classification
        total_samples = 0

        for batch in train_loader:
            labels = batch["label"]
            for label in labels:
                label_counts[label] += 1
                total_samples += 1

        # Calculate weights inversely proportional to class frequencies
        weights = total_samples / (2 * label_counts)  # 2 is number of classes
        return weights.to(self.model.device)

    def set_class_weights(self, weights):
        """Set class weights for the model's loss function."""
        # Convert tensor to regular Python list for JSON serialization
        weights_list = weights.cpu().tolist()
        self.model.config.class_weights = weights_list
        # Keep tensor version for computation
        self.class_weights_tensor = weights
