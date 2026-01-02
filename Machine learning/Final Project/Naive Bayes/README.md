# Naive Bayes Classifier Module

This directory contains the training, tuning, and inference logic for the Fake News Detection Naive Bayes classifier. It is designed to work seamlessly with the `Data preprocessing and cleanup` module.

## Features
1.  **Three Robust Algorithms**: Implements the mathematical foundations of Bernoulli, Multinomial, and Complement Naive Bayes from scratch.
2.  **Custom Hyperparameter Optimization**: Uses an exhaustive Grid Search with 5-Fold Cross-Validation.
3.  **Holistic Best Model Selection**: Automatically selects the winning model based on a multi-metric composite score.
4.  **High-Scale APIs with Analytics**: Integrated support for batch CSV processing with automated distribution reports (Real vs Fake percentages).
5.  **Dynamic Training Configuration**: Ability to specify custom hyperparameter search grids (JSON) for all variants via GUI or API.
6.  **Hot-Swap Configuration**: Modify parameters like `alpha` and `fit_prior` in real-time without retraining.

---

## How the "Best" Model is Selected
To ensure the chosen classifier is robust and not biased towards a single metric, the system uses a **Holistic Composite Scoring** condition.

**The Condition**:
The script evaluates four key metrics for every parameter combination:
- **F1-Score (Macro)**
- **Recall**
- **Accuracy**
- **Precision**

The **Composite Score** is the simple average of these four:
$$Composite = \frac{F1 + Recall + Accuracy + Precision}{4}$$

During training, the system compares all variants (MNB, CNB, BNB) and selects the one that maximizes this **Composite Score**. This ensures the winner has the most balanced performance profile for production use.

---

## API Reference (REST)

### 1. Single Prediction
`POST /predict`
```json
{
  "title": "Breaking Headline",
  "text": "Full article content...",
  "algorithm": "bnb",
  "alpha": 0.01,
  "fit_prior": true
}
```

### 2. Batch Prediction & Analytics
`POST /predict_batch`
*   **Body**: Form-data with `file` (CSV).
*   **Result**: 
    - `results`: List of predictions per headline.
    - `summary`: Analytics object including `real_percentage`, `fake_percentage`, and total counts.

### 3. Automated Training (Dynamic)
`POST /train`
*   **Body (Optional)**: JSON with `mnb_grid`, `cnb_grid`, `bnb_grid` objects.
*   **Action**: Triggers background retraining using the provided or default search spaces.

---

## Cross-Platform GUI Integration

### A. Python-Native GUIs (PyQt, Streamlit, etc.)
You can use the model classes directly for maximum performance.

```python
import joblib
from naive_bayes_model import BernoulliNB

# Load and Update
model = joblib.load("models/bnb.joblib")
model.update_params(alpha=0.1, fit_prior=True)

# Predict
X_vector = transform_records([title], [text], ...)
label = model.predict(X_vector)[0]
```

### B. Web & Mobile GUIs (React, Flutter, etc.)
Use standard HTTP requests to communicate with the Flask backend.

```javascript
async function analyze(article) {
  const response = await fetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title: article.title, text: article.text })
  });
  const data = await response.json();
}
```

---

## Usage
1. Ensure `Data preprocessing and cleanup/Output/features_out` exists.
2. Run training: `python train_tune_naive_bayes.py`.
3. Start GUI server: `python ../GUI/app.py`.
