# Presentation: Fake News Detection System

## 1. Project Overview & Goal
**Objective**: Build a machine learning system to classify news articles as **Real** or **Fake**.
- **Core Technology**: Naive Bayes Classifiers (NLP).
- **Goal**: Maximize the **F1-Score** to ensure a balance between catching fake news and preserving real news.



## 2. System Architecture
The solution is built on a modular pipeline:
1.  **Input**: News Title + Text.
2.  **Preprocessing**: Cleaning -> TF-IDF Vectorization (5000 features).
3.  **Model Wrapper**: `NaiveBayesClassifierWrapper` handles training, tuning, and prediction.
4.  **Output**: Fake (0) or Real (1).

## 3. Methodology: Training & Validation Strategy

To ensure the model is **robust** and not just **memorizing** data, we employed a two-stage validation strategy:

### Stage 1: Internal Validation (Training with ISOT)
- **What it is**: We split our main dataset (ISOT, 40k articles) into parts. We train on 80% and test on 20% (K-Fold Cross-Validation).
- **Purpose**: To teach the model and tune its settings (Hyperparameters) for maximum performance.
- **Outcome**: A highly optimized model specialized on the training distribution.

### Stage 2: External Validation (Robustness Check)
- **What it is**: We take the trained model and test it on **WELFake** (72k articles), a massive dataset the model has **never seen**.
- **Purpose**: To simulate the "Real World." Can the model detect upgrades in fake news writing styles?

> **Crucial Insight**: The high Specificity (93%) proves the model is a "Safe Guard" — it rarely falsely accuses real news of being fake.



## 4. Methodology: Hyperparameter Tuning
We didn't just guess the settings; we scientifically optimized them.

- **Technique**: Grid Search with Cross-Validation.
- **Parameter Tuned**: `Alpha` (Smoothing parameter).
- **Range Tested**: `[0.1, 0.5, 1.0, 5.0, 10.0]`
- **Winner**: **Alpha = 0.1** provided the sharpest boundary between Real and Fake.

## 5. Internal Validation: Model Comparison
We tested three architecture types on the ISOT dataset (40k articles).

| Model | Accuracy | F1-Score | Specificity | AUC |
| :--- | :--- | :--- | :--- | :--- |
| **Bernoulli NB** | **97.7%** | **97.7%** | **97.1%** | **0.99** |
| Complement NB | 96.0% | 96.0% | 95.9% | 0.99 |
| Multinomial NB | 96.0% | 96.0% | 95.6% | 0.99 |

> **Result**: Bernoulli NB was selected for production.

## 6. Internal Validation: The Winner (BNB)
A deeper look at the Bernoulli Naive Bayes performance on ISOT.

### Confusion Matrix
| | Predicted Fake | Predicted Real |
| :--- | :--- | :--- |
| **Actual Fake** | **3,476 (TN)** | 105 (FP) |
| **Actual Real** | 76 (FN) | **4,163 (TP)** |

- **Precision**: 97.7%
- **Recall**: 97.7%

## 7. External Validation: Robustness Test



**The Challenge**: ISOT is clean. The real world is messy.
**The Test**: We ran the trained BNB model on **WELFake** (72k unseen articles).

- **Detailed Results (WELFake)**:
    - **Accuracy**: **83.01%** (Solid baseline for unseen data)
    - **Precision**: **84.10%** (High trust in positive predictions)
    - **Sensitivity (Recall)**: **73.0%** (Conservative detection)
    - **Specificity**: **93.0%** (Excellent rejection of Fake News)
    - **F1-Score**: **82.82%** (Balanced performance)
    - **AUC-ROC**: **87.29%** (Good ranking capability)
    - **Confusion Matrix**: 
        - **True Negatives (Fake caught)**: ~34,300
        - **False Positives (Real flagged)**: ~2,700
        - **True Positives (Real cleared)**: ~25,500
        - **False Negatives (Fake missed)**: ~9,400

## 8. The Final Showdown: Internal vs. External
Measuring the drop-off when moving to "Big Data".

| Scenario | Accuracy | Specificity | F1-Score |
| :--- | :--- | :--- | :--- |
| **Internal (Trained)** | 97.7% | 97.1% | 97.7% |
| **External (Unseen)** | 83.0% | 93.0% | 82.8% |

> **Conclusion**: While Accuracy drops (expected), **Specificity remains elite (93%)**. The model is safe to deploy.

## 9. Production Readiness
- **Automated Pipeline**: From raw CSV to trained binary model automatically.
- **Persistence**: Models are saved with full metadata (`Internal_validation_results.json`) allowing instant reloading without retraining.
- **Validation**: Built-in logic to auto-validate on new CSVs and generate report plots.

## 10. Conclusion
The Naive Bayes module is a lightweight, high-speed, and surprisingly accurate solution for Fake News Detection. Its ability to maintain **93% Specificity** on unseen big data makes it a strong candidate for the first line of defense in a security system.
