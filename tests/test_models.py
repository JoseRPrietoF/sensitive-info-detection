import unittest
import torch
from transformers import AutoTokenizer
from src.models.classifier import SensitiveInformationClassifier


class TestSensitiveInformationClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that can be reused across tests."""
        cls.model_name = "prajjwal1/bert-tiny"
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.model = SensitiveInformationClassifier(
            model_name=cls.model_name, num_classes=2, num_frozen_layers=0
        )

    def test_model_initialization(self):
        """Test if model initializes correctly with expected configuration."""
        self.assertIsInstance(self.model, SensitiveInformationClassifier)
        self.assertEqual(self.model.model.config.num_labels, 2)
        self.assertEqual(self.model.model.config.model_type, "bert")

    def test_forward_pass(self):
        """Test if forward pass works with expected input shapes."""
        # Prepare sample input
        text = "Email server credentials: SMTP_USER=brucecurtis@[DOMAIN] SMTP_PASS=_#8NeiqSQe."
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )

        # Run forward pass
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )
        # Check output shapes and types
        self.assertIsInstance(outputs.logits, torch.Tensor)
        self.assertEqual(outputs.logits.shape[0], 1)  # batch size
        self.assertEqual(outputs.logits.shape[1], 2)  # num classes

    def test_compute_class_weights(self):
        """Test class weight computation."""

        # Create a mock DataLoader with imbalanced classes
        class MockDataset:
            def __init__(self):
                self.data = [
                    {"label": torch.tensor(0)},
                    {"label": torch.tensor(0)},
                    {"label": torch.tensor(1)},
                ]

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        mock_loader = torch.utils.data.DataLoader(MockDataset(), batch_size=1)

        weights = self.model.compute_class_weights(mock_loader)

        # Check weights shape and values
        self.assertEqual(len(weights), 2)
        self.assertGreater(
            weights[1], weights[0]
        )  # minority class should have higher weight

    def test_layer_freezing(self):
        """Test if layers are correctly frozen."""
        model = SensitiveInformationClassifier(
            model_name=self.model_name, num_classes=2, num_frozen_layers=1
        )

        # Check if embeddings are frozen
        for param in model.model.bert.embeddings.parameters():
            self.assertFalse(param.requires_grad)

        # Check if specified layers are frozen
        for param in model.model.bert.encoder.layer[0].parameters():
            self.assertFalse(param.requires_grad)

        # Check if remaining layers are trainable
        for param in model.model.bert.encoder.layer[-1].parameters():
            self.assertTrue(param.requires_grad)

    def test_model_prediction(self):
        """Test model predictions on sample inputs."""
        texts = [
            "My password is 12345",  # sensitive
            "The weather is nice today",  # not sensitive
        ]

        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        )

        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )
            predictions = torch.argmax(outputs.logits, dim=1)

        # Check prediction shape and type
        self.assertEqual(predictions.shape[0], 2)
        self.assertTrue(torch.is_tensor(predictions))


if __name__ == "__main__":
    unittest.main()
