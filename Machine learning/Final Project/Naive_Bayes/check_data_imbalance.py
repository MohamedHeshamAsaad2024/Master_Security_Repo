import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the Data_preprocessing_and_cleanup directory to the system path to import features_pipeline
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Data_preprocessing_and_cleanup')))
import features_pipeline

"""
    Checks for class imbalance in the training data used by the Naive Bayes models.
    
    This function performs the following steps:
    1. Loads the training labels (y_train) from the preprocessed features directory.
    2. Counts the number of samples for each class (Fake vs. Real).
    3. Calculates the percentage distribution.
    4. Prints a summary report to the console.
    5. Generates and saves a bar chart visualization of the class distribution.
"""
def check_imbalance():
    # Define the path to the features directory
    # Assuming the script is in Naive_Bayes/ and data is in Data_preprocessing_and_cleanup/Output/features_out
    # This matches the structure seen in test_nb.py
    features_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Data_preprocessing_and_cleanup', 'Output', 'features_out'))
    
    print(f"Loading data from: {features_dir}")
    
    if not os.path.exists(features_dir):
        print(f"Error: Features directory not found at {features_dir}")
        print("Please run the feature pipeline first to generate the data.")
        return

    # Load the training data
    # We only need y_train to check for imbalance
    try:
        _, _, y_train, _ = features_pipeline.load_feature_matrices(features_dir, scaled=False)
    except Exception as e:
        print(f"Error loading feature matrices: {e}")
        return

    # Calculate class counts
    total_samples = len(y_train)
    fake_count = np.sum(y_train == 0)
    real_count = np.sum(y_train == 1)
    
    fake_percent = (fake_count / total_samples) * 100
    real_percent = (real_count / total_samples) * 100

    print("-" * 30)
    print("Class Distribution (Training Data)")
    print("-" * 30)
    print(f"Total Samples: {total_samples}")
    print(f"Fake (0):      {fake_count} ({fake_percent:.2f}%)")
    print(f"Real (1):      {real_count} ({real_percent:.2f}%)")
    print("-" * 30)

    if fake_count == real_count:
        print("Result: Perfectly Balanced")
    elif abs(fake_percent - real_percent) < 10:
         print("Result: Roughly Balanced")
    else:
        print("Result: Imbalanced")

    # Generate Visualization
    output_dir = os.path.join(os.path.dirname(__file__), "Data_Analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    labels = ['Fake', 'Real']
    counts = [fake_count, real_count]
    colors = ['#ff9999', '#66b3ff'] # Redish for Fake, Blueish for Real

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, counts, color=colors, edgecolor='black')
    
    # Add counts on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}\n({height/total_samples*100:.1f}%)',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.title(f'Class Distribution in Training Data\n(Total: {total_samples})', fontsize=14)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.ylim(0, max(counts) * 1.15) # Add some headroom for text
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    output_path = os.path.join(output_dir, "class_distribution.png")
    plt.savefig(output_path)
    print(f"\nVisualization saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    check_imbalance()
