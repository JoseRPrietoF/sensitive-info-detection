"""
FastAPI application for serving the sensitive information detection model.
"""

from typing import Dict
import os
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow
from dotenv import load_dotenv
from src.data.utils import anonymize_text

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Sensitive Information Detection API",
    description="API for detecting sensitive information in text using a fine-tuned transformer model",
    version="1.0.0",
)

# Model configuration
MODEL_PATH = Path("models/best_model.pt")
MODEL_NAME = "prajjwal1/bert-tiny"  # Should match training configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
model = None


class PredictionRequest(BaseModel):
    """Request body for prediction endpoint."""

    text: str


class PredictionResponse(BaseModel):
    """Response body for prediction endpoint."""

    sensitive: bool
    confidence: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score of the prediction, ranging from 0.0 to 1.0",
    )


@app.on_event("startup")
async def load_model():
    """Load model and tokenizer on startup."""
    global model

    try:
        # Get MLflow run ID from environment
        run_id = os.getenv("MLFLOW_RUN_ID")
        if not run_id:
            raise RuntimeError("MLFLOW_RUN_ID environment variable not set")

        model = load_model_from_mlflow(run_id)

    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Sensitive Information Detection API"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> Dict:
    """
    Predict if text contains sensitive information.

    Args:
        request: Request body containing text to analyze

    Returns:
        Dictionary with prediction and confidence score
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Anonymize input text
        anonymized_text = anonymize_text(request.text)

        # Get prediction
        with torch.no_grad():
            outputs = model(anonymized_text)
            # [{'label': 'LABEL_0', 'score': 0.9354790449142456}]

            prediction = int(outputs[0]["label"].split("_")[1])
            confidence = outputs[0]["score"]

        return {"sensitive": bool(prediction), "confidence": confidence}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def load_model_from_mlflow(run_id: str):
    """Load model and tokenizer from MLflow."""
    # First try MLflow standard path
    logged_model = f"runs:/{run_id}/model"

    try:
        return mlflow.transformers.load_model(logged_model, return_type="pipeline")
    except Exception:
        # Search through mlruns subdirectories for the run_id
        # Only for the docker
        mlruns_path = Path("/app/mlruns")
        if mlruns_path.exists():
            for folder in mlruns_path.iterdir():
                if folder.is_dir():
                    run_path = folder / run_id
                    if run_path.exists():
                        model_path = (
                            f"/app/mlruns/{folder.name}/{run_id}/artifacts/model"
                        )
                        return mlflow.transformers.load_model(
                            model_path, return_type="pipeline"
                        )

        raise RuntimeError(
            f"Could not find model for run_id {run_id} in MLflow or local directories"
        )
