# Binary Text Classification for Sensitive Information Detection
## 🚀 Problem Statement

This project implements a **text classification system** to detect sensitive information (e.g., Social Security Numbers, credentials) in logs using fine-tuned Large Language Models (LLMs). This system helps ensure compliance and security in enterprise workflows.

---
## 🛠️ Setup & Installation

### Using Docker (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/JoseRPRietoF/sensitive-info-detection
cd sensitive-info-detection
```

2. Build the Docker image:
```bash
docker build -t sensitive-info-detection -f Dockerfile .
```

### Local Installation (Alternative)

1. Create a virtual environment with conda and activate it:
```bash
conda create -n sensitive-info python=3.11
conda activate sensitive-info
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

---
## 📊 Data Format

The model expects data in CSV format with the following structure:

```csv
text,sensitive_label
"Example text without sensitive info",0
"Password: abc123xyz",1
"Regular log message",0
"SSN: 123-45-6789",1
```

Fields:
- `text`: String containing the text to classify
- `sensitive_label`: Binary label (0 or 1)
  - 0: Non-sensitive information
  - 1: Contains sensitive information

Examples of sensitive information:
- Passwords and credentials
- Social Security Numbers
- Credit card numbers
- API keys and tokens
- Personal identifying information

Note: The actual training data is not included in this repository for privacy reasons. You'll need to provide your own dataset following this format.

---
## 🚀 Usage

### Data Analysis & Training Pipeline

Using Docker:
```bash
docker run --gpus all -it -v $(pwd)/:/app/ sensitive-info-detection bash launch_eda_train.sh
```

This script will:
1. Launch Jupyter notebook for data exploration and statistics
2. Clean and relabel the training data
3. Anonymize sensitive information
4. Start the model training process

### 📈 MLflow Integration

The training process is tracked using MLflow, which logs:
- Model parameters
- Training/validation metrics
- Model artifacts
- Learning curves

After training, start the MLflow server to view results:
```bash
mlflow server --host 127.0.0.1 --port 8080
```

Visit `http://localhost:8080` to see the training process and results of all models.

### Serving the Model API

First, set the MLflow run ID in `src/api/.env`:
```bash
MLFLOW_RUN_ID=YOUR_RUN_ID_HERE
```

Using Docker:
```bash
docker run -p 8000:8000 -v $(pwd)/:/app/ sensitive-info-detection uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Example API request:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Email server credentials: user@example.com password: example123"}'
```

Example response:
```json
{
    "sensitive": true,
    "confidence": 0.8996686935424805
}
```

## 📋 Tasks
### 1. Data Preprocessing & Augmentation Pipeline
- Implement a preprocessing pipeline to clean and prepare text data.
- Apply **data augmentation techniques** (e.g., synonym replacement, backtranslation) to improve model robustness.
### 2. Fine-Tuned LLM Model
- Fine-tune a base LLM on the provided dataset.
- Optimize hyperparameters to handle class imbalance and avoid overfitting.
### 3. Model Evaluation Report
- Provide a report with **precision, recall, F1-score, and ROC-AUC** (if applicable).
- Include analysis of potential bias, overfitting, and actionable improvement steps.
### 4. Inference API
- Build a simple API using **FastAPI** or **Flask** to serve the model.
- The API should accept text input and return a JSON response with the predicted class and confidence score.

---

## 📁 Project Structure
```
.
├── notebooks/
│ ├── __init__.py
│ └── EDA.ipynb
├── src/
│ ├── __init__.py
│ ├── analyze_data.py
│ ├── baseline.py
│ ├── train.py
│ ├── api/
│ │ ├── __init__.py
│ │ └── main.py
│ ├── data/
│ │ ├── __init__.py
│ │ ├── data_loader.py
│ │ └── utils.py
│ └── models/
│ ├── __init__.py
│ ├── classifier.py
│ └── trainer.py
├── tests/
│ ├── __init__.py
│ └── test_models.py
├── Dockerfile
├── README.md
└── requirements.txt
```

### Directory Structure Explanation

#### Main Directories
- **notebooks/** - Jupyter notebooks for exploratory data analysis and development
  - `EDA.ipynb` - Interactive notebook for data exploration
  - `EDA.py` - Python version of the notebook for reproducibility

- **src/** - Main source code directory
  - **api/** - FastAPI application code for model serving
  - **data/** - Data handling utilities and preprocessing
  - **models/** - Model architecture and training logic
  - `analyze_data.py` - Scripts for data analysis
  - `baseline.py` - Baseline model implementation
  - `train.py` - Main training script

- **tests/** - Unit tests directory
  - `test_models.py` - Tests for the BERT-based classifier model, including initialization, forward pass, class weights computation, layer freezing, and predictions

To run the tests, use the following command from the project root:
```bash
python -m unittest discover tests
```

## 📊 Model Architecture

The solution implements a robust text classification system with the following key components:

### 1. BERT-based Classification Model
- Uses a pre-trained transformer model (`prajjwal1/bert-tiny` by default) for efficient inference
- Implements layer freezing for efficient fine-tuning:
  - Embedding layers are frozen for stability
  - Configurable number of encoder layers can be frozen
- Binary classification head for sensitive/non-sensitive prediction
- Implements class-weighted loss function:
  - Addresses class imbalance in the dataset
  - Weights inversely proportional to class frequencies
  - Improves convergence speed and model performance on minority class
  - Helps prevent bias towards majority class predictions

### 2. Data Processing Pipeline
- **Text Anonymization**:
  - Automatic detection and replacement of sensitive patterns
  - Handles names, URLs, dates, SSNs, credit cards, and passwords
  - Uses regex-based pattern matching for consistent anonymization
- **Data Quality**:
  - TF-IDF similarity analysis to detect inconsistent labels
  - Majority voting for label correction
  - Clean and anonymized dataset versions

### 3. Training Features
- MLflow integration for experiment tracking:
  - Model parameters
  - Training/validation metrics
  - Model artifacts
  - Learning curves
- Early stopping based on validation F1 score
- Evaluation metrics:
  - Accuracy, Precision, Recall
  - F1-score
  - ROC-AUC

### 4. Production Deployment
- FastAPI-based serving:
  - Automatic text anonymization
  - Confidence score output
  - Error handling and validation
- Docker support for reproducible deployment
- MLflow model registry integration

## 📁 Provided Resources
- **Simulated Dataset**:
  - Columns: `text` (string), `sensitive_label` (0 or 1).
  - Examples:
    | `text`                                                          | `sensitive_label` |
    |----------------------------------------------------------------|-------------------|
    | "Employee SSN: 123-45-6789. Address: 123 Fake Street."         | 1                 |
    | "Update password policy for compliance with ISO 27001."        | 0                 |
---

## 📤 Deliverables
1. **Code**:
   - Preprocessing/augmentation scripts.
   - Training/evaluation code (Jupyter notebook or Python scripts).
   - API implementation.
2. **Trained Model**: Saved model weights and tokenizer.
3. **Evaluation Report**: PDF or Markdown file with metrics and analysis.
4. **API Documentation**: Instructions to run the API locally.

---

## 📬 Submission Guidelines

1. Clone this repository.
2. Add your code, model, and report to the repo.
3. Update this README including setup instructions (dependencies, API commands).
4. Push your changes to the repo. No need to keep everything in a separate branch.

---

## ❓ FAQs

**Q: Can I share the dataset externally?**
A: No—the simulated dataset is confidential. Use it only for this challenge.

**Q: What if I have more questions?**
A: Along with the invitation to this repository, you will also receive an invite to a Slack workspace. There, we will be available to answer any questions you may have about the challenge.

---
Good luck! We're excited to see your innovative solutions. 🎯
