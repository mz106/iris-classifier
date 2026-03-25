# Iris Classifier

A small machine learning project that trains two classifiers on the built-in Iris dataset from scikit-learn:

- Decision Tree Classifier
- K-Nearest Neighbors (KNN) Classifier

The project saves trained models to disk and prints basic evaluation metrics.

## Project Structure

```text
iris_classifier/
|- src/
|  |- train.py
|- tests/
|  |- _train.py
|- outputs/
|  |- model.joblib
|  |- model_knn.joblib
|- notebooks/
|  |- iris_model.ipynb
|- requirements.txt
|- README.md
```

## Features

- Trains a Decision Tree model on the Iris dataset
- Trains a KNN model on the Iris dataset
- Reports test accuracy for both models
- Displays a confusion matrix for the Decision Tree model
- Saves trained models with joblib in the outputs folder

## Requirements

- Python 3.10+
- Dependencies listed in requirements.txt

## Setup

1. Create a virtual environment:

```bash
python -m venv venv
```

2. Activate the virtual environment:

Windows (Git Bash):

```bash
source venv/Scripts/activate
```

Windows (PowerShell):

```powershell
.\venv\Scripts\Activate.ps1
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Training

Run the training script:

```bash
python src/train.py --test-size 0.2 --random-state 42
```

What it does:

- Loads the Iris dataset
- Splits data into train and test sets
- Trains Decision Tree and KNN models
- Prints accuracy values
- Prints and plots confusion matrix
- Saves model files to outputs/model.joblib and outputs/model_knn.joblib

## Run Tests (pytest)

Run the test file:

```bash
pytest tests/_train.py -q
```

The test checks that:

- The training script runs successfully
- Both model files are created
- Saved models can be loaded and used for prediction
- Both trained models achieve accuracy greater than 0.90 on the held-out test set

## Model Quality Expectation

The automated test suite asserts that both trained models achieve accuracy greater than 0.90 when run with:

```bash
python src/train.py --test-size 0.2 --random-state 42
```

## Notes

- The training script currently uses fixed values for test split and random seed.
- The confusion matrix plot uses matplotlib; on headless environments, tests set a non-interactive backend.
