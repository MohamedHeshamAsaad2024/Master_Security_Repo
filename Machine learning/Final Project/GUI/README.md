# Fake News Detector GUI

A modern, web-based interface for the Fake News Detection system. It allows users to input news headlines and articles to get real-time predictions with confidence scores and explainability features.

## Features

*   **Real-time Analysis**: Instant classification using the trained Naive Bayes model.
*   **Confidence Score**: Visual meter showing how certain the model is.
*   **Explainability**: Highlights the top words that influenced the decision (using model weights).
*   **Premium UI**: Glassmorphism design with responsiveness and animations.

## Prerequisites

Ensure you have the following installed (via `pip`):
*   `flask`
*   `joblib`
*   `numpy`
*   `scikit-learn`
*   `pandas`

## Usage

### 1. Start the Server
Navigate to this directory (`Machine learning/Final Project/GUI`) and run:

```bash
python app.py
```

You should see output indicating the server is running (usually at `http://127.0.0.1:5000`).

### 2. Open the Interface
Open your web browser and go to `http://127.0.0.1:5000`.

### 3. Analyze News
1.  Enter a **Headline** (e.g., "Aliens land in New York").
2.  (Optional) Enter the **Article Body**.
3.  Click **Generic Analysis**.
4.  View the result (Real vs Fake) and the "Key Indicators" section to see which words triggered the classification.

## Technical Details

*   **Backend**: Flask (Python). Use `app.py` to handle API requests.
*   **Model Loading**: The app automatically searches for the trained model in `../Naive Bayes/models/` and feature artifacts in `../Data preprocessing and cleanup/Output/features_out`.
*   **Inference**: It re-uses the exact same preprocessing pipeline (`feature_pipeline.transform_records`) to ensure consistency between training and deployment.

## Extending the GUI (Developer Guide)

The GUI is designed to support multiple classifiers. Currently, only **Naive Bayes** is implemented. Follow these steps to integrate Logistic Regression, SVM, or XGBoost.

### 1. Train & Save the Model
Train your new model (e.g., Logistic Regression) using the `Data preprocessing` pipeline and save it as a joblib file:
```python
joblib.dump(lr_model, "models/logistic_regression.joblib")
```

### 2. Update Backend (`app.py`)
Modify `load_resources` to load your new model file:
```python
# app.py
def load_resources():
    global model, lr_model # Add your global variable
    # ... existing code ...
    lr_model = joblib.load(os.path.join(MODEL_DIR, "logistic_regression.joblib"))
```

### 3. Handle Prediction Logic
Update the `predict` route in `app.py` to switch models based on the `model_type` parameter:

```python
# app.py > predict()
model_type = data.get('model_type', 'naive_bayes')

if model_type == 'naive_bayes':
    selected_model = model
elif model_type == 'logistic_regression':
    selected_model = lr_model  # Use your loaded model
# ... handle others ...

# Use selected_model for prediction
prediction_cls = selected_model.predict(X_input)[0]
```

### 4. Verify Frontend
Ensure `templates/index.html` has a corresponding option value that matches your python logic:
```html
<option value="logistic_regression">Logistic Regression</option>
```
