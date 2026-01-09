# Naive Bayes Integration Guide

This directory contains the Naive Bayes implementation for the Fake News Detection project. It supports three types of Naive Bayes models (BNB, CNB, MNB) and provides robust APIs for training, tuning, and prediction.

## APIs Overview

### `NaiveBayesClassifierWrapper(features_dir)`
Initializes the classifier and loads feature extraction artifacts.
- `features_dir`: Path to the directory containing `X_train_unscaled.npz`, `artifacts/`, etc.

### `train(param_grids, model_types=['BNB', 'CNB', 'MNB'])`
Trains and tunes the specified model types using 5-fold cross-validation.
- **Selection Logic**: Automatically chooses the "best" parameters and model type based on the **average of Accuracy, Precision, Recall, and F1-score**.
- **Input**: `param_grids` (e.g., `{'BNB': {'alpha': [0.1, 1.0]}, ...}`).
- **Returns**: Best parameters and all 4 metrics for each model.

### `predict_single(title, text, subject=None, model_type='best', parameters=None)`
Predicts the label for a single news item.
- **Flexibility**: You can choose any classifier (`BNB`, `CNB`, `MNB`) or use the `'best'`.
- **Custom Parameters**: If `parameters` are provided (e.g., `{'alpha': 2.0}`), the model is trained on-the-fly with these hyperparameters before prediction.
- **Returns**: `1` (Real) or `0` (Fake).

### `predict_csv(csv_path, model_type='best', parameters=None)`
Processes an entire CSV file for batch prediction.
- **Full Metrics**: If the CSV contains labels, it returns full metrics: Accuracy, Precision, Recall, and F1.
- **Returns**: A dictionary with `predictions` and `metrics`.

## Usage Example

```python
from Naive Bayes.naive_bayes_model import NaiveBayesClassifierWrapper

# Initialize
nb = NaiveBayesClassifierWrapper("path/to/features_out")

# 1. Train only BNB as an example
nb.train({'BNB': {'alpha': [0.1, 1.0]}}, model_types=['BNB'])

# 2. Predict with specific classifier and custom parameters
pred = nb.predict_single("Title", "Body text", model_type='MNB', parameters={'alpha': 2.0})

# 3. Batch predict a CSV and see full metrics
results = nb.predict_csv("path/to/WELFake_Dataset.csv", model_type='best')
print(results['metrics'])

# 4. Save and Load Models (Using .joblib)
save_dir = "path/to/save/models"

# Save: Creates .joblib files for models (e.g. nb_BNB.joblib) and metadata (nb_bestmodel_configuration.joblib)
nb.save_models(save_dir)

# Load: Restores trained models and the knowledge of which one was "best"
nb_new = NaiveBayesClassifierWrapper("path/to/features_out")
nb_new.load_models(save_dir)

# Now you can predict without retraining!
nb_new.predict_single("Another Title", "Text...", model_type='best')
```

### `save_models(save_dir)`
Saves the trained models and metadata to the specified directory.
- **Artifacts Created**:
  - `nb_BNB.joblib`, `nb_MNB.joblib`, etc.: The actual trained model files.
  - `nb_bestmodel_configuration.joblib`: Stores the "best" model type, best hyperparameters, and performance metrics.
    > **Why is this file needed?**
    > Without `metadata.joblib`, the system wouldn't remember which of the saved models (BNB, MNB, or CNB) performed best during training. This file acts as the "brain's memory," allowing `load_models()` to automatically select the optimal model and settings without requiring you to manually specify them or re-run the tuning process.

### `load_models(save_dir)`
Loads previously trained models and metadata.
- **Usage**: Call this after initializing the wrapper to restore the state of a trained system.
- **Effect**: You can immediately use `predict_single` or `predict_csv` using the `model_type='best'` from the previous run.

