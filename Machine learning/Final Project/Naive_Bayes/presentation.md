# Presentation: Fake News Detection System (Refined)

## 1. Project Overview
A machine learning system designed to classify news articles as **Real** or **Fake** using NLP and Naive Bayes classifiers (BNB, CNB, MNB).

## 2. Advanced Training Logic
- **Multi-Metric Selection**: The system automatically selects the best classifier based on the **average of Accuracy, Precision, Recall, and F1-score**.
- **Flexibility**: Supports training specific classifiers (e.g., training only Bernoulli NB).
- **On-the-fly Tuning**: Prediction APIs allow providing custom hyperparameters, triggering a just-in-time fit on the training data.

## 3. Results: ISOT Dataset (Internal)
| Model | Accuracy | Precision | Recall | F1-Score | Avg Score |
|---|---|---|---|---|---|
| **Bernoulli NB** | **97.69%** | **97.69%** | **97.69%** | **97.68%** | **97.69%** |
| Complement NB | 96.01% | 96.01% | 96.01% | 96.01% | 96.01% |

## 4. Evaluation: WELFake Dataset (External)
To test robustness, the model (BNB) was evaluated on the **WELFake dataset** (approx. 72,000 articles):
- **Accuracy**: 83.01%
- **F1-Score**: 82.82%
- **Precision**: 84.11%
- **Recall**: 83.01%

## 5. Graphical User Interface (GUI)
- **Multi-Tab Interface**: Separate spaces for Training, Single Prediction, and Batch Evaluation.
- **Extensible Design**: Dropdown support for Naive Bayes variants and placeholders for SVM, XGBoost, and Logistic Regression.
- **Dynamic Tuning**: Supports manual parameter input for real-time model adjustment.

## 6. Inference APIs
