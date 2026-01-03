# Fake News Detection GUI

This directory contains the Graphical User Interface (GUI) for the Fake News Detection project. The application provides a user-friendly way to train models, predict individual news veracity, and evaluate large datasets.

## Core Features

### 1. Global Model Selection
A persistent dropdown menu across all tabs allows you to select the active classifier:
- **Naive Bayes**: Bernoulli NB, Complement NB, Multinomial NB, or the "Best NB" (automatically selected based on the average performance of Accuracy, Precision, Recall, and F1).
- **Placeholders**: Support is already built into the UI for future integration of **SVM**, **XG Boost**, and **Logistic Regression**.

### 2. Tab-Based Functionality
- **Configuration**: Set the directory for preprocessed features and initialize the model wrappers.
- **Training**: Trigger 5-Fold Cross-Validation training with automated hyperparameter tuning. View detailed metrics (Accuracy, Precision, Recall, F1) in real-time.
- **Single Prediction**: Enter a news title and text to get an instant "Real" or "Fake" classification. Supports manual hyperparameter overrides for on-the-fly testing.
- **Batch Evaluation**: Load external CSV files (e.g., WELFake) to perform batch classification and generate performance reports.

## How to Run

Ensure you have the required dependencies installed (pandas, scikit-learn, joblib) and run the following command from the project root:

```bash
python "Machine learning/Final Project/GUI/main_gui.py"
```

## Integration with Naive Bayes
The GUI interacts directly with the `Naive_Bayes/naive_bayes_model.py` wrapper, which in turn uses the feature extraction pipeline from `Data_preprocessing_and_cleanup`.
