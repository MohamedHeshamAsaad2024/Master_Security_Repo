# Project Report: Fake News Detection via Naive Bayes (Refined)

## Introduction
This project implements a modular system for fake news classification. Recent refinements focused on enhancing the model selection logic and providing more flexible prediction APIs.

## Refined Implementation Details

### Multi-Metric Model Selection
Instead of relying solely on F1-score, the `train` API now identifies the best model by averaging four key performance indicators: **Accuracy, Precision, Recall, and F1-score**. This ensures the winning model (Bernoulli NB in our tests) is robust across all classification quality dimensions.

### Dynamic Prediction APIs
The prediction APIs (`predict_single`, `predict_csv`) were enhanced to support:
- **Type Selection**: Users can choose any available classifier (BNB, CNB, MNB).
- **On-the-fly Training**: Providing a `parameters` dictionary triggers a new training cycle with those specific hyperparameters, enabling rapid experimentation without manually calling `train`.

### Graphical User Interface (GUI)
To make the system accessible, a Tkinter-based GUI was developed in `GUI/main_gui.py`. It provides an intuitive interface for:
1. **Training**: Users can select the model type and features directory to trigger the training pipeline.
2. **Prediction**: A dedicated tab for classifying individual news articles with options for parameter override.
3. **Batch Analysis**: A tab for evaluating large CSV datasets (e.g., WELFake) and visualizing comparative metrics.
The GUI is designed with an extensible architecture, allowing future classification algorithms to be "plugged in" via the centralized model selection dropdown.

## Results and Evaluation

### Internal Performance (ISOT Dataset)
| Model Type | Accuracy | Precision | Recall | F1 Score | Best Alpha |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Bernoulli NB** | **0.9769** | **0.9769** | **0.9769** | **0.9768** | **0.1** |
| Complement NB | 0.9601 | 0.9601 | 0.9601 | 0.9601 | 0.1 |

### External Performance (WELFake Dataset)
The system was tested against the external **WELFake** dataset to measure generalization performance:
- **Accuracy**: 0.8301
- **F1 Score**: 0.8282
- **Avg Performance**: ~83%

The drop in performance compared to the internal dataset is expected but still demonstrates strong generalization (83% accuracy) on completely unseen data sources.

## Conclusion
The refined Naive Bayes system provides a highly flexible and analytically sound approach to fake news detection, suitable for both research tuning and real-world inference.
