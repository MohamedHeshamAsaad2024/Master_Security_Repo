# Logistic Regression for Fake News Detection

A complete implementation of logistic regression from scratch using gradient descent optimization with NumPy. Successfully trained on the ISOT fake news dataset achieving **99.36% accuracy** with optimized hyperparameters.

## Features

- **Pure NumPy implementation** - No scikit-learn dependency for the core algorithm
- **Sparse matrix support** - Efficiently handles TF-IDF features
- **Gradient Descent optimization** - Efficient parameter learning
- **L2 Regularization** - Prevents overfitting
- **Binary Classification** - Predicts fake vs. real news with probabilities
- **Hyperparameter Tuning** - Grid search with configurable parameters
- **Comprehensive Visualizations** - Confusion matrix, ROC curve, learning curve, feature importance
- **Configuration System** - JSON-based hyperparameter management
- **Production-ready** - Optimized model achieving 99.36% test accuracy

## Project Structure

```
Logistic_regression/
├── logistic_regression.py          # Core LogisticRegression class
├── utils.py                         # Evaluation metrics and utilities
├── train_fake_news.py              # Training script (uses config.json)
├── test_on_welfake.py              # External dataset testing script
├── hyperparameter_tuning.py        # Grid search for optimal hyperparameters
├── config.json                      # Hyperparameter configuration
├── trained_fake_news_model.pkl     # Optimized model (99.36% accuracy)
├── Visualizations_Out/              # Auto-generated visualizations
└── README.md                        # This file
```

## Trained Model Performance

The included pre-trained model was trained on the ISOT fake news dataset with optimized hyperparameters:

- **Dataset**: 31,276 training articles, 7,820 test articles
- **Features**: 20,000 TF-IDF features
- **Test Accuracy**: **99.36%**
- **Precision**: 99.32%
- **Recall**: 99.50%
- **Sensitivity**: 99.50%
- **Specificity**: 99.19%
- **F1-Score**: 99.41%
- **AUC-ROC**: 0.9997

### Optimized Hyperparameters

The model uses the following optimized hyperparameters (found via grid search):

```json
{
  "learning_rate": 0.05,
  "n_iterations": 1500,
  "regularization": 0.001
}
```

## Installation

This implementation requires NumPy, SciPy, pandas, matplotlib, seaborn, and joblib:

```bash
pip install numpy scipy pandas matplotlib seaborn joblib
```

## Quick Start

### 1. Training with Optimized Hyperparameters

```bash
python train_fake_news.py
```

This will:
- Load hyperparameters from `config.json`
- Train the model on ISOT dataset
- Generate 6 visualizations in `Visualizations_Out/`
- Save the trained model as `trained_fake_news_model.pkl`

### 2. Testing on External Dataset

```bash
python test_on_welfake.py
```

Tests the trained model on WELFake dataset and generates visualizations showing cross-dataset generalization.

### 3. Hyperparameter Tuning

```bash
python hyperparameter_tuning.py
```

Runs grid search over 64 hyperparameter combinations to find optimal settings:
- Tests learning rates: [0.01, 0.05, 0.1, 0.2]
- Tests iterations: [500, 1000, 1500, 2000]
- Tests regularization: [0.001, 0.01, 0.1, 1.0]
- Generates heatmaps and performance visualizations
- Optionally updates `config.json` with best parameters

## Configuration System

All scripts now use `config.json` for centralized hyperparameter management.

### config.json Structure

```json
{
  "hyperparameters": {
    "learning_rate": 0.05,
    "n_iterations": 1500,
    "regularization": 0.001,
    "threshold": 0.5
  },
  "hyperparameter_search_space": {
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "n_iterations": [500, 1000, 1500, 2000],
    "regularization": [0.001, 0.01, 0.1, 1.0]
  }
}
```

### Hyperparameter Explanations

**Learning Rate** (0.01 - 0.2)
- **Lower (0.01-0.05)**: Slower but more stable convergence
- **Higher (0.1-0.2)**: Faster but may overshoot optimal values
- **Recommended**: 0.05 - 0.1

**Number of Iterations** (500 - 2000)
- **Lower (500-1000)**: Faster training, may underfit
- **Higher (1500-2000)**: Better convergence, slower training
- **Recommended**: 1000 - 1500

**Regularization/L2** (0.001 - 1.0)
- **Lower (0.001-0.01)**: Less regularization, more complex model
- **Higher (0.1-1.0)**: More regularization, simpler model, prevents overfitting
- **Recommended**: 0.01 - 0.1

## Visualizations

The training script automatically generates 6 visualizations:

### ISOT Training Results
Saved in `Visualizations_Out/`:
1. **confusion_matrix.png** - TP, TN, FP, FN breakdown
2. **learning_curve.png** - Cost reduction over iterations
3. **feature_importance.png** - Top words for real vs fake news
4. **roc_curve.png** - ROC curve with AUC score
5. **precision_recall_curve.png** - Precision-Recall trade-off
6. **metrics_comparison.png** - Training vs Test metrics

### WELFake Test Results
Saved in `Visualizations_Out/WELFake_Test/`:
- Cross-dataset evaluation visualizations (84.18% accuracy on WELFake)

### Hyperparameter Tuning Results
Saved in `Visualizations_Out/Hyperparameter_Tuning/`:
- Heatmaps showing hyperparameter interactions
- Bar chart of top 10 configurations

## API Reference

### LogisticRegression Class

#### Constructor Parameters

- `learning_rate` (float, default=0.01): Learning rate for gradient descent
- `n_iterations` (int, default=1000): Number of training iterations
- `regularization` (float, default=0.0): L2 regularization parameter (lambda)
- `verbose` (bool, default=False): If True, print cost during training

#### Methods

**`fit(X, y)`**
- Train the model using gradient descent
- **Parameters:**
  - `X`: array-like or sparse matrix, shape (n_samples, n_features)
  - `y`: array-like, shape (n_samples,) - Target values (0 or 1)
- **Returns:** self

**`predict(X, threshold=0.5)`**
- Predict class labels
- **Parameters:**
  - `X`: array-like or sparse matrix
  - `threshold`: float - Classification threshold
- **Returns:** array of predicted labels (0 or 1)

**`predict_proba(X)`**
- Predict class probabilities
- **Parameters:**
  - `X`: array-like or sparse matrix
- **Returns:** array of probabilities for the positive class

**`score(X, y)`**
- Calculate accuracy score
- **Returns:** float - Accuracy score

### Utility Functions

**`calculate_metrics(y_true, y_pred, y_pred_proba=None)`**
- Calculate accuracy, precision, recall, sensitivity, specificity, F1-score, AUC-ROC, and confusion matrix

**`print_metrics(metrics)`**
- Print classification metrics in a readable format

## Usage Examples

### Example 1: Using Pre-trained Model

```python
import joblib
from scipy.sparse import load_npz

# Load pre-trained optimized model
model = joblib.load('trained_fake_news_model.pkl')

# Load your TF-IDF features
X_test = load_npz('path/to/your/tfidf_features.npz')

# Make predictions
predictions = model.predict(X_test)  # 0 = fake, 1 = real
probabilities = model.predict_proba(X_test)

# Example output
if predictions[0] == 0:
    print(f"FAKE NEWS (confidence: {1-probabilities[0]:.2%})")
else:
    print(f"REAL NEWS (confidence: {probabilities[0]:.2%})")
```

### Example 2: Training with Custom Hyperparameters

Edit `config.json`:
```json
{
  "hyperparameters": {
    "learning_rate": 0.05,
    "n_iterations": 1500,
    "regularization": 0.1
  }
}
```

Then run:
```bash
python train_fake_news.py
```

### Example 3: Finding Optimal Hyperparameters

```bash
# Run grid search
python hyperparameter_tuning.py

# When prompted, type 'yes' to update config.json with best params

# Train final model with optimized hyperparameters
python train_fake_news.py
```

## Workflow

1. **Initial Training**: Run `train_fake_news.py` with default config
2. **Hyperparameter Tuning**: Run `hyperparameter_tuning.py` to find best params
3. **Update Config**: Accept the prompt to update config.json
4. **Final Training**: Run `train_fake_news.py` again with optimized params
5. **External Testing**: Run `test_on_welfake.py` to test generalization

## Algorithm Details

### Sigmoid Function
```
σ(z) = 1 / (1 + e^(-z))
```

### Cost Function (Binary Cross-Entropy with L2 Regularization)
```
J(w,b) = -1/m * Σ[y*log(ŷ) + (1-y)*log(1-ŷ)] + (λ/2m)*Σ(w²)
```

### Gradient Descent Update Rules
```
w = w - α * dJ/dw
b = b - α * dJ/db
```

Where:
- `α` is the learning rate
- `m` is the number of training samples
- `λ` is the regularization parameter

## Dataset Information

### ISOT Fake News Dataset (Training)
- **Source**: University of Victoria, ISOT Research Lab
- **Content**: Political news articles from 2016-2017
- **Fake news sources**: Various unreliable websites
- **Real news sources**: Reuters.com (verified journalism)
- **Total articles**: ~44,000 (after preprocessing)

### WELFake Dataset (External Testing)
- Used for cross-dataset validation
- Achieves 84.18% accuracy (demonstrates good generalization)

## Tips for Best Results

1. **Feature Preprocessing**: Always use the same TF-IDF vectorizer
   - Use `load_artifacts()` from feature_pipeline
   - Use `transform_records()` for new data

2. **Hyperparameter Tuning**: 
   - Start with grid search to find optimal values
   - Use validation split (20%) for tuning
   - Monitor learning curve for convergence

3. **Threshold Adjustment**:
   - Higher threshold (0.7-0.9): Fewer false positives
   - Lower threshold (0.3-0.5): Fewer false negatives
   - Default (0.5): Balanced performance

4. **Performance Optimization**:
   - Use sparse matrices for memory efficiency
   - Batch predictions for multiple articles
   - Cache the loaded model

5. **Model Limitations**:
   - Trained on political news from 2016-2017
   - Requires English text
   - Performance may vary on short articles or social media posts

## Advanced Features

### Cross-Dataset Validation
```bash
python test_on_welfake.py
```
Tests model generalization on external WELFake dataset.

### Feature Importance Analysis
Automatically generated visualization showing:
- Top 20 words predicting **REAL** news
- Top 20 words predicting **FAKE** news

### Metrics Comparison
Side-by-side visualization of training vs test metrics to identify overfitting.

## Troubleshooting

**Issue**: Low accuracy on new data
- **Solution**: Ensure same preprocessing pipeline is used (TF-IDF vectorizer, scaling)

**Issue**: Model not converging
- **Solution**: Lower learning rate or increase iterations in config.json

**Issue**: Overfitting (train acc >> test acc)
- **Solution**: Increase regularization parameter

