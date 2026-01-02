# Naive Bayes Classifier Module

This directory contains the training, tuning, and inference logic for the Fake News Detection Naive Bayes classifier. It is designed to work seamlessly with the `Data preprocessing and cleanup` module.

## Features
1.  **Three Robust Algorithms**: Implements the mathematical foundations of:
    *   **Multinomial Naive Bayes**: Optimized for frequency-based text classification.
    *   **Complement Naive Bayes**: Enhanced for handling data imbalances.
    *   **Bernoulli Naive Bayes**: Specialized for binary presence/absence patterns.
2.  **Custom Hyperparameter Optimization**: Uses an exhaustive Grid Search with 5-Fold Cross-Validation to rigorously test thousands of parameter combinations ($\alpha$, smoothing priors, normalization).
3.  **Automatic Best Model Selection**: The script evaluates F1-Score, Precision, and Recall for all variants and automatically promotes the best performing architecture (BernoulliNB).
4.  **High Performance**: Achieves **~97.8% Accuracy** on the test set.

> **Note on Multinomial vs. Complement NB**:
> You may notice that `MultinomialNB` and `ComplementNB` yield nearly identical results. This is expected behavior for **Binary Classification**.
> *   In a 2-class problem (Fake vs Real), the "Complement" of Class A is exactly Class B.
> *   Therefore, both algorithms calculate probabilities based on the exact same underlying word counts, just using different formulations.
> *   `ComplementNB` typically diverges and shows its true strength in **imbalanced multi-class** problems.

## Usage

### Prerequisites
Ensure the `Data preprocessing and cleanup` module has been run and `Output/features_out` exists.

### Running the Script
Run the script from the project root or this directory:

```bash
python "Machine learning/Final Project/Naive Bayes/train_tune_naive_bayes.py"
```

### Outputs
The script will output:
1.  **Console Logs**: Detailed training progress, best parameters found, and evaluation metrics.
2.  **Trained Model**: Saved to `models/best_naive_bayes.joblib`.
3.  **Feature Importance**: Prints the top 10 keywords driving "Fake" and "Real" classifications.

## Integration / API

To use the trained model in other applications (like the GUI):

```python
import joblib
from feature_pipeline import transform_records, FeatureConfig, load_artifacts

# 1. Load Model and Preprocessing Artifacts
model = joblib.load("models/best_naive_bayes.joblib")
artifacts = load_artifacts("../Data preprocessing and cleanup/Output/features_out")
config = FeatureConfig(**json.load(open("../Data preprocessing and cleanup/Output/features_out/artifacts/config.json")))

# 2. Prepare Input
title = "Breaking News"
text = "Some fake content here..."

# 3. Transform & Predict
# NOTE: The model pipeline handles Feature Selection (SelectKBest) automatically.
# We just need to vectorize the text using the original vocabulary.
X_new = transform_records([title], [text], None, artifacts, config, scaled=False)
prediction = model.predict(X_new)[0] # 0 = Fake, 1 = Real
probabilities = model.predict_proba(X_new)[0]

print(f"Prediction: {prediction}, Confidence: {max(probabilities)}")
```
