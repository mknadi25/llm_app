# LLM App - Machine Learning Project Classifier

A production-ready machine learning system for classifying machine learning projects by their tags (e.g., "natural-language-processing", "computer-vision") based on project titles and descriptions. Built with PyTorch, Ray, and MLflow for distributed training, hyperparameter tuning, and model serving.

## Table of Contents

- [Project Description](#project-description)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Architecture Overview](#architecture-overview)
- [Installation & Setup](#installation--setup)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
  - [Training](#training)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
  - [Evaluation](#evaluation)
  - [Serving](#serving)
  - [Prediction](#prediction)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [API Documentation](#api-documentation)
- [Testing](#testing)
- [Deployment](#deployment)
- [Development Workflow](#development-workflow)
- [Results & Artifacts](#results--artifacts)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Project Description

This project implements an end-to-end machine learning system that automatically categorizes machine learning projects into predefined tags based on their titles and descriptions. The system uses a fine-tuned BERT-based model (SciBERT) for multi-class text classification.

### Use Cases

- **Project Organization**: Automatically tag and categorize ML projects in repositories
- **Content Discovery**: Help users discover projects by category
- **Analytics**: Analyze trends in ML project types over time
- **Recommendation Systems**: Build recommendation engines based on project categories

### Key Capabilities

- **Distributed Training**: Scale training across multiple workers with Ray Train
- **Automated Hyperparameter Tuning**: Optimize model performance with Ray Tune
- **Model Serving**: Deploy models as REST APIs with Ray Serve
- **Experiment Tracking**: Track experiments, metrics, and models with MLflow
- **Slice-based Evaluation**: Evaluate model performance on specific data slices

## Features

- ✅ **Multi-class Text Classification**: Classify projects into multiple categories
- ✅ **Distributed Training**: Train models across multiple CPUs/GPUs with Ray Train
- ✅ **Hyperparameter Tuning**: Automated hyperparameter optimization with Ray Tune and HyperOpt
- ✅ **Model Serving API**: RESTful API for real-time predictions using FastAPI and Ray Serve
- ✅ **MLflow Integration**: Complete experiment tracking and model registry
- ✅ **Slice-based Evaluation**: Evaluate performance on specific data subsets (e.g., NLP projects with LLMs)
- ✅ **Data Validation**: Comprehensive testing for data quality and model behavior
- ✅ **Docker Support**: Containerized deployment for easy scaling

## Technology Stack

### Core ML Frameworks
- **PyTorch** (2.0.0): Deep learning framework
- **Transformers** (4.28.1): Pre-trained models (SciBERT)
- **scikit-learn** (1.2.2): Machine learning utilities

### Distributed Computing
- **Ray** (2.7.0): Distributed computing framework
  - **Ray Train**: Distributed model training
  - **Ray Tune**: Hyperparameter optimization
  - **Ray Serve**: Model serving
  - **Ray Data**: Distributed data processing

### MLOps & Infrastructure
- **MLflow** (2.3.1): Experiment tracking and model registry
- **FastAPI** (0.95.2): Modern web framework for APIs
- **Docker**: Containerization

### Data Processing
- **pandas** (2.0.1): Data manipulation
- **numpy** (1.24.3): Numerical computing

### Testing & Quality
- **pytest** (7.3.1): Testing framework
- **black** (23.3.0): Code formatting
- **flake8** (6.0.0): Linting
- **isort** (5.12.0): Import sorting

## Architecture Overview

```
┌─────────────┐
│   Dataset   │
│   (CSV)     │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  Data Loading   │
│  & Preprocessing│
└──────┬──────────┘
       │
       ├─────────────────┐
       │                 │
       ▼                 ▼
┌─────────────┐   ┌──────────────┐
│   Training  │   │ Hyperparameter│
│  (Ray Train)│   │    Tuning     │
│             │   │  (Ray Tune)   │
└──────┬──────┘   └──────┬───────┘
       │                 │
       └────────┬────────┘
                │
                ▼
         ┌──────────────┐
         │   MLflow     │
         │ Model Registry│
         └──────┬───────┘
                │
                ▼
         ┌──────────────┐
         │  Evaluation  │
         │  & Metrics   │
         └──────┬───────┘
                │
                ▼
         ┌──────────────┐
         │   Serving    │
         │ (Ray Serve)  │
         │   FastAPI    │
         └──────────────┘
```

### Data Flow

1. **Data Loading**: Load and preprocess CSV datasets using Ray Data
2. **Training**: Distributed training with automatic checkpointing
3. **Tuning**: Hyperparameter optimization with early stopping
4. **Tracking**: All experiments logged to MLflow
5. **Evaluation**: Comprehensive metrics and slice analysis
6. **Serving**: Deploy best model as REST API

## Installation & Setup

### Prerequisites

- Python 3.10+
- pip
- (Optional) CUDA-capable GPU for faster training
- (Optional) Docker for containerized deployment

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd llm_app
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Step 4: Set Environment Variables

```bash
export GITHUB_USERNAME="your_username"
export S3_ENDPOINT="URL_address_to_s3_bucket"  # Optional, for S3 storage
```

### Step 5: Configure Dataset Location

Edit `config/config.yaml` to specify your dataset location:

```yaml
data:
  dataset_loc: "https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/dataset.csv"
```

### Step 6: Initialize Ray

Ray will be initialized automatically when you run training/serving commands, but you can also initialize it manually:

```bash
ray start --head
```

## Quick Start

### Train a Model

```bash
python -m src.train \
    --experiment-name "my_experiment" \
    --dataset-loc "https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/dataset.csv" \
    --train-loop-config '{"dropout_p": 0.5, "lr": 1e-4, "lr_factor": 0.8, "lr_patience": 3}' \
    --num-workers 2 \
    --cpu-per-worker 2 \
    --num-epochs 10 \
    --batch-size 256
```

### Make a Prediction

```bash
python -m src.predict \
    --run-id "your_run_id" \
    --title "Neural Network for Image Classification" \
    --description "A deep learning model using CNN architecture"
```

### Start the API Server

```bash
python -m src.serve --run-id "your_run_id" --threshold 0.9
```

## Usage Guide

### Training

Train a model with distributed Ray Train:

```bash
python -m src.train \
    --experiment-name <experiment_name> \
    --dataset-loc <dataset_url_or_path> \
    --train-loop-config <json_config> \
    --num-workers <num_workers> \
    --cpu-per-worker <cpus> \
    --gpu-per-worker <gpus> \
    --num-epochs <epochs> \
    --batch-size <batch_size> \
    --results-fp <results_filepath>
```

**Example:**

```bash
python -m src.train \
    --experiment-name "llm_job" \
    --dataset-loc "datasets/full train dataset.csv" \
    --train-loop-config '{"dropout_p": 0.5, "lr": 1e-4, "lr_factor": 0.8, "lr_patience": 3}' \
    --num-workers 2 \
    --cpu-per-worker 2 \
    --gpu-per-worker 0 \
    --num-epochs 10 \
    --batch-size 256 \
    --results-fp "results/training_results.json"
```

**Parameters:**
- `experiment-name`: Name for MLflow experiment
- `dataset-loc`: Path or URL to CSV dataset
- `train-loop-config`: JSON string with hyperparameters (dropout_p, lr, lr_factor, lr_patience)
- `num-workers`: Number of Ray workers
- `cpu-per-worker`: CPUs per worker
- `gpu-per-worker`: GPUs per worker (0 for CPU-only)
- `num-epochs`: Training epochs
- `batch-size`: Batch size
- `results-fp`: Path to save training results

### Hyperparameter Tuning

Run hyperparameter optimization with Ray Tune:

```bash
python -m src.tune \
    --experiment-name <experiment_name> \
    --dataset-loc <dataset_url_or_path> \
    --initial-params <json_config> \
    --num-workers <num_workers> \
    --num-runs <num_trials> \
    --num-epochs <epochs> \
    --batch-size <batch_size> \
    --results-fp <results_filepath>
```

**Example:**

```bash
python -m src.tune \
    --experiment-name "tuning_experiment" \
    --dataset-loc "datasets/full train dataset.csv" \
    --initial-params '[{"dropout_p": 0.5, "lr": 1e-4, "lr_factor": 0.8, "lr_patience": 3}]' \
    --num-workers 1 \
    --num-runs 10 \
    --num-epochs 5 \
    --batch-size 256 \
    --results-fp "results/tuning_results.json"
```

**Tuning Space:**
- `dropout_p`: Uniform(0.3, 0.9)
- `lr`: LogUniform(1e-5, 5e-4)
- `lr_factor`: Uniform(0.1, 0.9)
- `lr_patience`: Uniform(1, 10)

### Evaluation

Evaluate a trained model on a holdout dataset:

```bash
python -m src.evaluate \
    --run-id <run_id> \
    --dataset-loc <holdout_dataset> \
    --results-fp <results_filepath>
```

**Example:**

```bash
python -m src.evaluate \
    --run-id "94ef9aab9c334010af5aaf553cb3f3d2" \
    --dataset-loc "datasets/test dataset.csv" \
    --results-fp "results/evaluation_results.json"
```

**Output Metrics:**
- Overall: precision, recall, F1 (weighted)
- Per-class: precision, recall, F1, sample count
- Slices: Performance on specific data subsets (e.g., NLP+LLM projects, short text)

### Serving

Start the model serving API:

```bash
python -m src.serve --run-id <run_id> [--threshold <threshold>]
```

**Example:**

```bash
python -m src.serve --run-id "94ef9aab9c334010af5aaf553cb3f3d2" --threshold 0.9
```

The API will be available at `http://localhost:8000` by default.

### Prediction

#### Command Line

```bash
python -m src.predict \
    --run-id <run_id> \
    --title <project_title> \
    --description <project_description>
```

**Example:**

```bash
python -m src.predict \
    --run-id "94ef9aab9c334010af5aaf553cb3f3d2" \
    --title "BERT for Sentiment Analysis" \
    --description "Fine-tuned BERT model for analyzing sentiment in text"
```

#### API Request

```bash
curl -X POST "http://localhost:8000/predict/" \
    -H "Content-Type: application/json" \
    -d '{
        "title": "BERT for Sentiment Analysis",
        "description": "Fine-tuned BERT model for analyzing sentiment in text"
    }'
```

## Project Structure

```
llm_app/
├── config/
│   └── config.yaml              # Configuration file
├── datasets/                     # Dataset files
│   ├── full train dataset.csv
│   ├── test dataset.csv
│   ├── tags.csv
│   └── combine_tags.py
├── deploy/
│   ├── jobs/
│   │   └── workloads.sh         # Full workflow script
│   └── services/
│       └── serve_model.py
├── efs/
│   └── mlflow/                  # MLflow model registry
├── logs/                        # Application logs
├── models/                      # Saved models
├── notebooks/
│   └── monitoring.ipynb         # Monitoring notebook
├── results/                     # Training/evaluation results
│   ├── training_results.json
│   ├── evaluation_results.json
│   └── run_id.txt
├── src/                         # Source code
│   ├── __init__.py
│   ├── config.py               # Configuration and logging
│   ├── data.py                 # Data loading and preprocessing
│   ├── models.py               # Model architecture
│   ├── train.py                # Training script
│   ├── tune.py                 # Hyperparameter tuning
│   ├── evaluate.py             # Evaluation script
│   ├── predict.py              # Prediction utilities
│   ├── serve.py                # API serving
│   └── utils.py                # Utility functions
├── tests/                       # Test suite
│   ├── code/                   # Code unit tests
│   ├── data/                   # Data validation tests
│   └── model/                  # Model behavior tests
├── Dockerfile                  # Docker configuration
├── Makefile                    # Development commands
├── pyproject.toml              # Development tool config
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Configuration

### Configuration File

Edit `config/config.yaml` to configure dataset location:

```yaml
data:
  dataset_loc: "https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/dataset.csv"
```

### Environment Variables

- `GITHUB_USERNAME`: Your GitHub username (required for Ray runtime)
- `S3_ENDPOINT`: S3 endpoint URL (optional, for artifact storage)

### MLflow Configuration

MLflow tracking URI is automatically set to `file://efs/mlflow/`. Models and experiments are stored in the `efs/mlflow/` directory.

## API Documentation

The serving API provides the following endpoints:

### Health Check

**GET** `/`

Check if the API is running.

**Response:**
```json
{
  "message": "OK",
  "status-code": 200,
  "data": {}
}
```

### Get Run ID

**GET** `/run_id/`

Get the run ID of the currently loaded model.

**Response:**
```json
{
  "run_id": "94ef9aab9c334010af5aaf553cb3f3d2"
}
```

### Predict

**POST** `/predict/`

Make predictions on project titles and descriptions.

**Request:**
```json
{
  "title": "Neural Network for Image Classification",
  "description": "A deep learning model using CNN architecture"
}
```

**Response:**
```json
{
  "results": [
    {
      "prediction": "computer-vision",
      "probabilities": {
        "computer-vision": 0.85,
        "natural-language-processing": 0.10,
        "other": 0.05
      }
    }
  ]
}
```

**Note:** If the prediction probability is below the threshold (default 0.9), the prediction will be set to "other".

### Evaluate

**POST** `/evaluate/`

Evaluate the model on a dataset.

**Request:**
```json
{
  "dataset": "https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/holdout.csv"
}
```

**Response:**
```json
{
  "results": {
    "timestamp": "January 28, 2025 02:30:00 PM",
    "run_id": "94ef9aab9c334010af5aaf553cb3f3d2",
    "overall": {
      "precision": 0.92,
      "recall": 0.91,
      "f1": 0.91,
      "num_samples": 1000.0
    },
    "per_class": {
      "computer-vision": {
        "precision": 0.95,
        "recall": 0.93,
        "f1": 0.94,
        "num_samples": 250.0
      }
    },
    "slices": {
      "nlp_llm": {
        "precision": 0.88,
        "recall": 0.87,
        "f1": 0.87,
        "num_samples": 50.0
      }
    }
  }
}
```

## Testing

### Run All Tests

```bash
python -m pytest tests/ --verbose
```

### Run Specific Test Categories

**Code Tests:**
```bash
python -m pytest tests/code/ --verbose
```

**Data Tests:**
```bash
python -m pytest tests/data/ --verbose --dataset-loc="datasets/full train dataset.csv"
```

**Model Tests:**
```bash
python -m pytest tests/model/ --verbose --run-id="your_run_id"
```

### Run Full Workflow

Execute the complete workflow script:

```bash
bash deploy/jobs/workloads.sh
```

This script runs:
1. Data validation tests
2. Code unit tests
3. Model training
4. Model evaluation
5. Model behavior tests
6. Artifact upload to S3 (if configured)

You can control which stages run using environment variables:

```bash
RUN_TEST_DATA=true \
RUN_TEST_CODE=true \
RUN_TRAIN=true \
RUN_EVALUATE=true \
RUN_TEST_MODEL=true \
RUN_SAVE_ARTIFACTS=false \
bash deploy/jobs/workloads.sh
```

## Deployment

### Docker Deployment

Build the Docker image:

```bash
docker build -t llm-app:latest .
```

Run the container:

```bash
docker run -p 8000:8000 \
    -e GITHUB_USERNAME="your_username" \
    llm-app:latest \
    --run-id "your_run_id" \
    --threshold 0.9
```

The API will be available at `http://localhost:8000`.

### Ray Cluster Deployment

For production deployment on a Ray cluster:

1. **Start Ray Cluster:**
```bash
ray start --head --port=6379
```

2. **Deploy Model:**
```bash
python -m src.serve --run-id "your_run_id"
```

3. **Scale Workers:**
```bash
ray start --address="<head-node-ip>:6379"
```

### Cloud Deployment

The system supports deployment on cloud platforms:

- **AWS**: Use EFS for MLflow storage, S3 for artifacts
- **GCP**: Use Cloud Storage for model registry
- **Azure**: Use Azure Blob Storage

Configure the `S3_ENDPOINT` environment variable for S3-compatible storage.

## Development Workflow

### Code Style

The project uses:
- **black**: Code formatting
- **flake8**: Linting
- **isort**: Import sorting

**Format code:**
```bash
make style
```

Or manually:
```bash
black .
flake8
python -m isort .
```

### Clean Project

Remove cache files and temporary files:

```bash
make clean
```

### Project Configuration

Development tools are configured in `pyproject.toml`:
- Black: 88 character line length
- Flake8: E501, W503, E203, E226 ignored
- Pytest: Tests in `tests/` directory

## Results & Artifacts

### Training Results

Training results are saved to `results/training_results.json`:

```json
{
  "timestamp": "January 28, 2025 02:30:00 PM",
  "run_id": "94ef9aab9c334010af5aaf553cb3f3d2",
  "params": {
    "dropout_p": 0.5,
    "lr": 0.0001,
    "lr_factor": 0.8,
    "lr_patience": 3
  },
  "metrics": [
    {
      "epoch": 0,
      "train_loss": 0.85,
      "val_loss": 0.82
    }
  ]
}
```

### Evaluation Results

Evaluation results are saved to `results/evaluation_results.json` with overall, per-class, and slice metrics.

### Model Registry

Trained models are stored in `efs/mlflow/` and tracked in MLflow. Access models via:

```python
import mlflow
mlflow.set_tracking_uri("file://efs/mlflow/")
run = mlflow.get_run("run_id")
```

### Run ID

The run ID for the best model is saved to `results/run_id.txt` after training.

## Troubleshooting

### Common Issues

**Ray initialization errors:**
- Ensure `GITHUB_USERNAME` environment variable is set
- Check that Ray is properly installed: `pip install ray[default]`

**MLflow tracking errors:**
- Verify `efs/mlflow/` directory exists and is writable
- Check MLflow tracking URI in `src/config.py`

**CUDA/GPU errors:**
- Set `--gpu-per-worker 0` for CPU-only training
- Verify CUDA installation if using GPUs

**Import errors:**
- Ensure virtual environment is activated
- Install dependencies: `pip install -r requirements.txt`
- Set PYTHONPATH: `export PYTHONPATH=$PYTHONPATH:$PWD`

**Dataset loading errors:**
- Verify dataset URL/path in `config/config.yaml`
- Check network connectivity for remote datasets
- Ensure CSV format matches expected schema (title, description, tag columns)

### Debugging Tips

1. **Check logs:**
   - Application logs: `logs/info.log` and `logs/error.log`
   - Ray logs: Check Ray dashboard or console output

2. **Verify configuration:**
   - Check `config/config.yaml` for dataset location
   - Verify environment variables are set

3. **Test components individually:**
   - Test data loading: `python -m pytest tests/data/`
   - Test model: `python -m pytest tests/model/ --run-id="your_run_id"`

4. **Ray dashboard:**
   - Access Ray dashboard at `http://localhost:8265` to monitor training

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Dataset**: [Made-With-ML](https://github.com/GokuMohandas/Made-With-ML) dataset for training and evaluation
- **Model Base**: [SciBERT](https://github.com/allenai/scibert) (allenai/scibert_scivocab_uncased) for the pre-trained transformer model
- **Framework**: Built with [Ray](https://www.ray.io/) for distributed computing and [MLflow](https://mlflow.org/) for experiment tracking
