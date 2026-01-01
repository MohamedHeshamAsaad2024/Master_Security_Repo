# Logistic Regression for Fake News Detection

A complete implementation of logistic regression from scratch using gradient descent optimization with NumPy. Successfully trained on the ISOT fake news dataset achieving **99.36% accuracy**.

## Features

- ✅ **Pure NumPy implementation** - No scikit-learn dependency for the core algorithm
- ✅ **Sparse matrix support** - Efficiently handles TF-IDF features
- ✅ **Gradient Descent optimization** - Efficient parameter learning
- ✅ **L2 Regularization** - Prevents overfitting
- ✅ **Binary Classification** - Predicts fake vs. real news with probabilities
- ✅ **Production-ready** - Trained model achieving 99.36% test accuracy
- ✅ **Comprehensive utilities** - Evaluation metrics and model persistence

## Project Structure

```
Logistic_regression/
├── logistic_regression.py          # Core LogisticRegression class with sparse matrix support
├── utils.py                         # Utility functions for evaluation and metrics
├── train_fake_news.py              # Training script for fake news detection
├── trained_fake_news_model.pkl     # Pre-trained model (99.36% accuracy)
└── README.md                        # This file
```

## Trained Model Performance

The included pre-trained model was trained on the ISOT fake news dataset:

- **Dataset**: 31,276 training articles, 7,820 test articles
- **Features**: 20,000 TF-IDF features
- **Test Accuracy**: **99.36%**
- **Precision**: 99.32%
- **Recall**: 99.50%
- **F1-Score**: 99.41%

## Installation

This implementation requires NumPy, SciPy, pandas, and joblib:

```bash
pip install numpy scipy pandas joblib
```

## Quick Start

### Using the Pre-trained Model

Load and use the pre-trained fake news detection model:

```python
import joblib
from scipy.sparse import load_npz

# Load the pre-trained model
model = joblib.load('trained_fake_news_model.pkl')

# Load your TF-IDF features (must use the same preprocessing pipeline)
X_test = load_npz('path/to/your/tfidf_features.npz')

# Make predictions
predictions = model.predict(X_test)  # 0 = fake news, 1 = real news
probabilities = model.predict_proba(X_test)  # confidence scores

# Example: Predict a single article
if predictions[0] == 0:
    print(f"FAKE NEWS (confidence: {1-probabilities[0]:.2%})")
else:
    print(f"REAL NEWS (confidence: {probabilities[0]:.2%})")
```

### Training from Scratch

Train a new model on the ISOT dataset:

```bash
python train_fake_news.py
```

This will:
1. Load preprocessed TF-IDF features from the ISOT dataset
2. Train the logistic regression model with gradient descent
3. Evaluate on both training and test sets
4. Display detailed metrics and confusion matrix
5. Save the trained model as `trained_fake_news_model.pkl`

**Expected output**: ~99% test accuracy after 1,000 iterations

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
  - `X`: array-like, shape (n_samples, n_features) - Training data
  - `y`: array-like, shape (n_samples,) - Target values (0 or 1)
- **Returns:** self

**`predict(X, threshold=0.5)`**
- Predict class labels
- **Parameters:**
  - `X`: array-like, shape (n_samples, n_features) - Samples to predict
  - `threshold`: float - Classification threshold
- **Returns:** array of predicted labels (0 or 1)

**`predict_proba(X)`**
- Predict class probabilities
- **Parameters:**
  - `X`: array-like, shape (n_samples, n_features) - Samples to predict
- **Returns:** array of probabilities for the positive class

**`score(X, y)`**
- Calculate accuracy score
- **Parameters:**
  - `X`: array-like, shape (n_samples, n_features) - Test samples
  - `y`: array-like, shape (n_samples,) - True labels
- **Returns:** float - Accuracy score

### Utility Functions

**`normalize_features(X, method='standardize')`**
- Normalize features using standardization or min-max scaling
- Returns normalized data and normalization parameters

**`calculate_metrics(y_true, y_pred)`**
- Calculate accuracy, precision, recall, F1-score, and confusion matrix

**`train_test_split(X, y, test_size=0.2, random_state=None)`**
- Split data into training and testing sets

**`print_metrics(metrics)`**
- Print classification metrics in a readable format

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

## Examples

### Example 1: Load and Use Pre-trained Model

```python
import joblib
from scipy.sparse import load_npz

# Load pre-trained model
model = joblib.load('trained_fake_news_model.pkl')

# Load your preprocessed features
X_test = load_npz('../Data_preprocessing_and_cleanup/Output/features_out/X_test_scaled.npz')

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Analyze results
fake_count = (predictions == 0).sum()
real_count = (predictions == 1).sum()
print(f"Fake news detected: {fake_count}")
print(f"Real news detected: {real_count}")
```

### Example 2: Training with Custom Hyperparameters

```python
from logistic_regression import LogisticRegression
from scipy.sparse import load_npz
import pandas as pd

# Load preprocessed data
X_train = load_npz('path/to/X_train_scaled.npz')
y_train = pd.read_csv('path/to/y_train.csv')['label'].values

# Train with custom settings
model = LogisticRegression(
    learning_rate=0.05,      # Lower learning rate for stability
    n_iterations=2000,       # More iterations for convergence
    regularization=0.1,      # Stronger regularization
    verbose=True             # Show training progress
)
model.fit(X_train, y_train)

# Save your model
import joblib
joblib.dump(model, 'my_custom_model.pkl')
```

### Example 3: Batch Prediction with Confidence Filtering

```python
import joblib
import numpy as np

model = joblib.load('trained_fake_news_model.pkl')

# Get predictions and probabilities
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Filter high-confidence predictions only
high_confidence_threshold = 0.90
confident_mask = (probabilities > high_confidence_threshold) | (probabilities < 0.10)

print(f"High confidence predictions: {confident_mask.sum()}/{len(predictions)}")
print(f"Accuracy on high-confidence: {np.mean(predictions[confident_mask] == y_test[confident_mask]):.2%}")
```

## Tips for Fake News Detection

1. **Feature Preprocessing**: Always use the same TF-IDF vectorizer used during training
   - The model expects exactly 20,000 features in the same order
   - Load the `tfidf.joblib` artifact from the features pipeline

2. **Text Cleaning**: Apply the same preprocessing steps:
   - Lowercase conversion
   - URL and email removal
   - Whitespace normalization
   
3. **Threshold Tuning**: Adjust based on your needs:
   - **Higher threshold (0.7-0.9)**: Fewer false positives (stricter fake news detection)
   - **Lower threshold (0.3-0.5)**: Fewer false negatives (catch more potential fake news)
   - **Default (0.5)**: Balanced performance

4. **Performance Optimization**:
   - Use sparse matrices for TF-IDF features (saves memory)
   - Batch predictions for multiple articles
   - Cache the loaded model for repeated use

5. **Model Limitations**:
   - Trained on political news from 2016-2017 (may not generalize to all topics)
   - Requires English text
   - Performance may vary on very short articles or social media posts

## Mathematical Background

Logistic regression models the probability of binary outcomes using the sigmoid function:

**Sigmoid Function:**
```
σ(z) = 1 / (1 + e^(-z))
```

**Cost Function (Binary Cross-Entropy with L2 Regularization):**
```
J(w,b) = -1/m * Σ[y*log(ŷ) + (1-y)*log(1-ŷ)] + (λ/2m)*Σ(w²)
```

**Gradient Descent Updates:**
```
w = w - α * dJ/dw
b = b - α * dJ/db
```

For this fake news detection task:
- **Input**: 20,000 TF-IDF features per article
- **Output**: Probability the article is real news (sigmoid output)
- **Decision**: Classify as fake (0) if probability < 0.5, real (1) otherwise

## Dataset Information

This model was trained on the **ISOT Fake News Dataset**:
- **Source**: University of Victoria, ISOT Research Lab
- **Content**: Political news articles from 2016-2017
- **Fake news sources**: Various unreliable websites
- **Real news sources**: Reuters.com (verified journalism)
- **Total articles**: ~44,000 (after preprocessing)

## Future Improvements

Potential enhancements for this implementation:
- Multi-class classification for news categories
- Mini-batch gradient descent for faster training
- Cross-dataset validation (test on other fake news datasets)
- Feature importance analysis (which words indicate fake news)
- Early stopping based on validation performance
- Support for newer BERT/transformer embeddings instead of TF-IDF
