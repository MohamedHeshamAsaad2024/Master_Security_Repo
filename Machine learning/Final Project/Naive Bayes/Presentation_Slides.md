# Presentation: Naive Bayes Implementation (Person 3)

---

## Slide 1: Title & Introduction
**Project Module: Naive Bayes Classifier**
*   **Role**: Person 3 (Model Implementation & Tuning)
*   **Objective**: Develop a fast, explainable, and accurate baseline model for Fake News Detection.

---

## Slide 2: Methodology & Data Flow
**Integration with Central Pipeline**
*   **Input Data**:Utilized unscaled TF-IDF Sparse Matrices (`X_train_unscaled.npz`).
*   **Why Unscaled?**: 
    *   Naive Bayes (Multinomial/Bernoulli) relies on feature counts/frequencies.
    *   Standard scaling (z-score) produces negative values, which breaks these algorithms.
*   **Direct Training**:
    *   Models are trained directly on the sparse matrices.
    *   Efficiency: Eliminates redundant re-processing steps.

---

## Slide 3: Optimization Strategy
**Hyperparameter Tuning with GridSearchCV**
*   **Technique**: 5-Fold Cross-Validation.
*   **Feature Strategy**:
    *   Utilized the full high-dimensional feature set (20,000 words) provided by the upstream preprocessing.
    *   No local feature reduction was necessary to achieve high performance.
*   **Model Variants Tested**:
    1.  **MultinomialNB**: The standard for text classification.
    2.  **ComplementNB**: Stronger for imbalanced datasets.
    3.  **BernoulliNB**: Binary "presence/absence" model.

---

## Slide 4: Experimental Results
**Best Performing Model: BernoulliNB**
*   **Configuration**:
    *   `alpha`: 0.01 (Low smoothing).
    *   `features`: Full Feature Set (20,000 features).
*   **Key Metrics**:
    *   **Accuracy**: **97.94%** (Test Set).
    *   **F1-Score**: 98%.
*   **Confusion Matrix**:
    *   Correctly identified **3,491** Fake articles.
    *   False Negatives: Only 90 articles missed.


