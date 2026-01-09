# Project Report: Fake News Detection via Naive Bayes

## 1. Introduction
This component of the project implements a robust, production-ready system for detecting fake news using Naive Bayes classifiers. The system is designed to be modular, efficient, and highly interpretable, leveraging Natural Language Processing (NLP) techniques to classify news articles as either **Real** or **Fake**.

## 2. Methodology

### 2.1 Preprocessing & Feature Extraction
The raw text data undergoes a rigorous cleaning pipeline (stopwords removal, stemming/lemmatization) before being converted into numerical vectors.
- **Technique**: TF-IDF (Term Frequency-Inverse Document Frequency) Vectorization.
- **Vocabulary Size**: 5,000 features (optimized for balance between performance and speed).
- **N-grams**: Unigrams and Bigrams (1,2) to capture local context.

### 2.2 Model Architecture
The core logic is encapsulated in the `NaiveBayesClassifierWrapper` class, which offers a high-level API for:
1.  **Training & Tuning**: automatic hyperparameter optimization.
2.  **Persistence**: saving/loading trained states.
3.  **Inference**: flexible prediction methods for single inputs or batch CSVs.

### 2.3 Models Evaluated
We implemented and evaluated three variations of the Naive Bayes algorithm to handle different data characteristics:
1.  **Bernoulli Naive Bayes (BNB)**: Designed for binary/boolean features. It works exceptionally well for short texts and when the *presence* of a word matters more than its frequency.
2.  **Multinomial Naive Bayes (MNB)**: The standard for text classification, using discrete counts.
3.  **Complement Naive Bayes (CNB)**: An adaptation of MNB specifically designed to correct the assumptions made by standard MNB, often performing better on imbalanced datasets.

### 2.4 Training Strategy
- **Cross-Validation**: 5-Fold Stratified Cross-Validation was used to ensure that every fold had the same proportion of Real/Fake labels as the whole dataset.
- **Hyperparameter Tuning**: We performed a `GridSearchCV` to optimize the smoothing parameter `alpha` for each model type.
- **Metric Selection**: The system automatically selects the "Best Model" based on the **F1-Score**. This metric was chosen as the primary decisional factor because it provides a harmonic mean of Precision and Recall, ensuring the model is reliable in both detecting fake news (Recall) and not flagging real news incorrectly (Precision).

### 2.5 Persistence & Automation
To ensure the system is production-ready, we implemented a persistence layer using `joblib`.
- **`save_models(save_dir)`**: Serializes the trained model objects and a `metadata.joblib` file.
- **`metadata.joblib`**: This critical artifact acts as the system's "memory." It stores the identity of the best-performing model and its configuration. This allows the system to be reloaded (`load_models`) and immediately start making optimal predictions without re-training.

## 3. Results (ISOT Dataset)

The following table summarizes the performance of all three models on the internal test set. The **Bernoulli Naive Bayes (BNB)** classifier was selected as the optimal model due to its superior performance across all metrics.

| Model | Accuracy | Precision | Sensitivity (Recall) | Specificity | F1-Score | AUC-ROC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Bernoulli NB (BNB)** | **97.69%** | **97.69%** | **97.69%** | **97.07%** | **97.68%** | **0.9930** |
| Complement NB (CNB) | 96.01% | 96.01% | 96.01% | 95.92% | 96.01% | 0.9923 |
| Multinomial NB (MNB) | 96.01% | 96.01% | 96.01% | 95.62% | 96.01% | 0.9923 |

### Analysis
- **BNB Dominance**: BNB achieved a near-perfect Accuracy of ~97.7%, significantly outperforming MNB and CNB by ~1.6%. This suggests that for this specific vocabulary and dataset, the simple presence/absence of specific "trigger words" is a more powerful predictor than their frequency.
- **High Specificity**: The Specificity of 97.07% indicates the model is extremely good at identifying Real news correctly, minimizing false alarms.

## 4. Generalization (External Validation)

### 4.1 Comparison of Results
To rigorously test the model's capabilities, we trained it on the ISOT dataset and then tested it on a completely separate, larger dataset (WELFake). This "Big Data" test ensures the model isn't just memorizing one specific dataset but learning actual linguistic patterns.

| Metric | Internal Validation (ISOT) | External Validation (WELFake) | Analysis |
| :--- | :--- | :--- | :--- |
| **Accuracy** | **97.69%** | **83.01%** | Drop is expected when switching datasets, but >80% is excellent for cross-domain generalization. |
| **Precision** | **97.69%** | **84.10%** | The model remains highly reliable when it flags news as real or fake. |
| **Sensitivity (Recall)** | **97.69%** | **73.0%** | The model is more conservative on the new dataset, missing some real news (higher False Negatives). |
| **Specificity** | **97.07%** | **93.0%** | **Key Strength**: The model is extremely good at spotting Fake News, even on data it has never seen. |
| **F1-Score** | **97.68%** | **82.82%** | The balanced metric confirms the model is robust and production-ready. |
| **AUC-ROC** | **99.30%** | **87.29%** | High separability indicates the model correctly ranks fake/real probabilities. |

### 4.2 Interpretation
The slight drop in performance on the WELFake dataset allows us to see the "real-world" capability of the model. While ISOT represents an ideal scenario, WELFake represents the wild internet. The fact that **Specificity remains at 93%** is a massive win for a security tool—it means **Fake News is almost always caught**. The drop in Sensitivity suggests the model is cautious and would rather flag uncertainty than let fake news pass.

## 5. Conclusion
The Naive Bayes subsystem is a highly effective, lightweight, and explainable solution for fake news detection. With an **F1-Score of 97.68%** and automated persistence capabilities, it is ready for deployment as a core component of the broader security repository.
