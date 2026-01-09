# Presentation: Fake News Detection System

## 1. Project Overview & Goal
**Objective**: Build a machine learning system to classify news articles as **Real** or **Fake**.
- **Core Technology**: Naive Bayes Classifiers (NLP).
- **Goal**: Maximize the **F1-Score** to ensure a balance between catching fake news and preserving real news.

### Generalization (External Validation - WELFake)
- **Accuracy**: **83.01%** (Strong performance on unseen data)
- **F1-Score**: **82.82%** (Consistent reliability)
- **Specificity**: **~93%** (Extremely good at blocking Fake News)
- **Sensitivity**: **~73%** (Reasonable detection of Real News)
- **Conclusion**: The model generalizes well, maintaining ~83% accuracy even on a completely different dataset, with a particular strength in identifying fake content.

## 2. System Architecture
The solution is built on a modular pipeline:
1.  **Input**: News Title + Text.
2.  **Preprocessing**: Cleaning -> TF-IDF Vectorization (5000 features).
3.  **Model Wrapper**: `NaiveBayesClassifierWrapper` handles training, tuning, and prediction.
4.  **Output**: Fake (0) or Real (1).

## 3. Methodology & Training
**Step 1: Training (Internal Validation)**
- **Dataset**: ISOT (40,000+ Articles).
- **Technique**: 5-Fold Cross-Validation with Grid Search.
- **Goal**: Tune hyperparameters (Alpha) to maximize F1-Score.
- **Result**: Bernoulli NB (BNB) emerged as the winner over Multinomial and Complement NB.

**Step 2: Testing "Big Data" (External Validation)**
- **Challenge**: Can the model survive in the wild?
- **Dataset**: WELFake (72,000+ Articles) - completely unseen during training.
- **Result**: The model successfully generalized, proving it didn't just memorize the training data.

## 4. Performance & Results

### Internal vs. External Battle
We stress-tested the model to see how it holds up.

| Metric | Internal (ISOT) | External (WELFake) |
| :--- | :--- | :--- |
| **Accuracy** | **~97.7%** | **~83.0%** |
| **F1-Score** | **~97.7%** | **~82.8%** |
| **Specificity** | **~97.1%** | **~93.0%** |
| **AUC-ROC** | **~99.3%** | **~87.3%** |

### Key Takeaway
- **Specificity is King**: On the external Big Data test, the model maintained a **93% Specificity**.
- **Security Implications**: This means the system is exceptionally good at blocking Fake News (True Negatives). It rarely lets a fake article pass through as real.
- **Robustness**: A drop from 97% to 83% is normal when switching data sources, but retaining >80% accuracy proves the model is robust and effective.

## 5. Production Readiness
- **Automated Pipeline**: From raw CSV to trained binary model automatically.
- **Persistence**: Models are saved with full metadata (`nb_metadata.json`) allowing instant reloading without retraining.
- **Validation**: Built-in logic to auto-validate on new CSVs and generate report plots.

## 6. Conclusion
The Naive Bayes module is a lightweight, high-speed, and surprisingly accurate solution for Fake News Detection. Its ability to maintain **93% Specificity** on unseen big data makes it a strong candidate for the first line of defense in a security system.
