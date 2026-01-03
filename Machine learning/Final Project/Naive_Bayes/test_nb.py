# Step 1: Import the necessary tools
import os
from naive_bayes_model import NaiveBayesClassifierWrapper

def test_nb_flow():
    """
    This function demonstrates how to use the NaiveBayesClassifierWrapper class.
    It simulates a complete 'Run' of the project.
    """
    
    # Step 2: Define the paths to your data
    # 'features_dir' is where the cleaned numeric data is stored
    features_dir = r"c:\Master\Repos\Master_Security_Repo\Machine learning\Final Project\Data_preprocessing_and_cleanup\Output\features_out"
    # 'save_dir' is where we will save the 'Finished Brain' (the model)
    save_dir = r"c:\Master\Repos\Master_Security_Repo\Machine learning\Final Project\Naive_Bayes\Models"
    # 'external_csv' is a completely different file to see if our AI can handle new data
    external_csv = r"c:\Master\Repos\Master_Security_Repo\Machine learning\Final Project\Data_preprocessing_and_cleanup\External_Datasets\WELFake_Dataset.csv"
    
    # Step 3: Create the 'Classifier' object
    # This automatically loads the vocabulary and configuration
    nb = NaiveBayesClassifierWrapper(features_dir)
    
    # --- EXAMPLE 1: Training ---
    # We tell the computer to try all models with different 'knobs' (alpha)
    print("--- 1. Training and Tuning BNB, CNB, and MNB ---")
    param_grids = {
        'BNB': {'alpha': [0.1, 0.5, 1.0, 5.0, 10.0]},
        'CNB': {'alpha': [0.1, 0.5, 1.0, 5.0, 10.0]},
        'MNB': {'alpha': [0.1, 0.5, 1.0, 5.0, 10.0]}
    }
    # This is the step where the model 'Learns' from the training data
    # We train all types to compare them
    best_params, metrics = nb.train(param_grids, model_types=['BNB', 'CNB', 'MNB'])
    
    # --- EXAMPLE 2: Single Prediction ---
    # This simulates you typing a news article into a website or GUI
    print("\n--- 2. Predicting with MNB and custom parameters (alpha=2.0) ---")
    title = "Trump says Russia probe is 'witch hunt'"
    text = "WASHINGTON (Reuters) - U.S. President Donald Trump on Thursday again called the investigation..."
    
    # We ask the model: "Based on what you learned, is this real or fake?"
    pred = nb.predict_single(title, text, model_type='MNB', parameters={'alpha': 2.0})
    
    # Print a human-readable result
    print(f"Prediction result: {'Real' if pred == 1 else 'Fake'} ({pred})")
    
    # --- EXAMPLE 3: External Evaluation ---
    # This checks if the AI is truly smart or just memorizing its own dataset
    if os.path.exists(external_csv):
        print(f"\n--- 3. Evaluating on External CSV: {os.path.basename(external_csv)} ---")
        # We use the trained BNB model to predict thousands of articles from the other file
        result = nb.predict_csv(external_csv, model_type='BNB')
        
        print("\nFinal Results for WELFake:")
        if 'metrics' in result:
            m = result['metrics']
            # Accuracy = Percentage of correct guesses
            print(f"Accuracy:  {m['accuracy']:.4f}")
            # F1 Score = Balance between being right and catching everything
            print(f"F1 Score:  {m['f1']:.4f}")
        else:
            print("No labels found in CSV, only predictions generated.")
    else:
        print(f"\nExternal CSV not found at: {external_csv}")

    # Step 4: Save the work
    # This writes the finished AI model to the 'Models' folder so you can use it later
    nb.save_models(save_dir)

# This part tells Python to run the function above when the script starts
if __name__ == "__main__":
    test_nb_flow()
