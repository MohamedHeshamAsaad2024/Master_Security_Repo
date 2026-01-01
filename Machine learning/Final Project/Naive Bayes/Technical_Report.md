# Technical Report: Fake News Detection using Naive Bayes

**Prepared by:** Person 3 (Naive Bayes Implementation)

## 1. Introduction

The objective of this module is to implement a robust **Naive Bayes Classifier** for the classification of Fake vs. Real news. This model acts as one of the core engines in our multi-model ensemble project.

The implementation focuses on:
1.  **High Efficiency**: Naive Bayes is computationally inexpensive and fast for inference.
2.  **Robustness**: Using standardized sparse feature matrices from the central feature extraction module.
3.  **Explainability**: Extracting highly weighted words to provide "Key Indicators" for the user.

## 2. Methodology & Technical Implementation

### 2.1. Feature Integration
Instead of raw text processing, this module integrates directly with the **Centralized Feature Pipeline**.
*   **Input**: TF-IDF Sparse Matrices (`X_train_unscaled.npz`, `X_test_unscaled.npz`).
*   **Why Unscaled?**: Naive Bayes models (Multinomial/Bernoulli) work natively with frequency counts or TF-IDF weights. Standardization (scaling to mean=0, std=1) allows negative values, which are incompatible with `MultinomialNB` and can destroy the sparsity benefit.

### 2.2. Model Training Strategy
We train the Naive Bayes models **directly** on the feature matrices provided by the centralized pipeline.

*   **No Secondary Pipeline**: Unlike complex architectures that might require local feature selection steps (e.g., `SelectKBest`), we established that Naive Bayes performs robustly on the full high-dimensional processed dataset (20,000 features).
*   **Leakage Prevention By Design**: Since we use the pre-split `X_train` and `X_test` matrices directly, there is zero risk of data leakage. Feature extraction statistics (TF-IDF means/variances) were strictly derived from the training set in the upstream process.

### 2.3. Hyperparameter Optimization strategy
We employed `GridSearchCV` with **5-Fold Cross-Validation** to exhaustively search for the optimal configuration.

**Search Space:**
We experimented with three distinct Naive Bayes variants:

1.  **MultinomialNB**:
    *   *Suitability*: Standard for text classification using word counts.
    *   *Params*: `alpha` (Smoothing), `fit_prior`.
2.  **ComplementNB**:
    *   *Suitability*: Optimized for imbalanced datasets.
    *   *Params*: `alpha`, `norm` (Weight normalization).
3.  **BernoulliNB**:
    *   *Suitability*: Binary models (Word present/absent).
    *   *Params*: `binarize` (Threshold).

**Optimization Objective**: `F1-Macro` Score (Balances Precision and Recall equally).

## 3. Results & Evaluation

### 3.1. Best Model Configuration
After training, the exhaustive search identified **BernoulliNB** as the top-performing variant.

*   **Model**: `BernoulliNB`
*   **Alpha (Smoothing)**: `0.01` (Low smoothing indicates high confidence in data).
*   **Features**: Utilized all available features produced by the preprocessing module.

### 3.2. Quantitative Metrics
The model was evaluated on the held-out Test Set (20% split).

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | **97.94%** | Extremely high overall correctness. |
| **Precision (Fake)** | 98% | When it predicts "Fake", it is almost always correct. |
| **Recall (Fake)** | 97% | It successfully catches 97% of all Fake News. |
| **F1-Score** | 98% | Excellent balance between precision and recall. |

*Confusion Matrix:*
```
[[3491   90]  <-- Fake News (Correct: 3491, Missed: 90)
 [  71 4168]] <-- Real News (False Alarm: 71, Correct: 4168)
```


