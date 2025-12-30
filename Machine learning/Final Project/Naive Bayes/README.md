# Naive Bayes Classifier Module

This directory contains the training, tuning, and inference logic for the Fake News Detection Naive Bayes classifier. It is designed to work seamlessly with the `Data preprocessing and cleanup` module.

## Files

*   **`train_tune_naive_bayes.py`**: The main script. It performs:
    1.  **Data Loading**: Imports preprocessed sparse matrices from the upstream pipeline.
    2.  **Hyperparameter Tuning**: Uses `GridSearchCV` to find optimal parameters for:
        *   Model Variants (`MultinomialNB`, `ComplementNB`, `BernoulliNB`).
        *   Feature Selection (`SelectKBest` with Chi-Squared test).
        *   Smoothing (`alpha`) and other model-specific hyperparameters.
    3.  **Evaluation**: Reports Accuracy, Precision, Recall, F1-Score, and Confusion Matrix on the test set.
    4.  **Feature Analysis**: Extracts the most indicative words for "Real" and "Fake" news for interpretability.
    5.  **Serialization**: Saves the best performing model to `models/best_naive_bayes.joblib`.

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
