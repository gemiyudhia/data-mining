from scripts.preprocess_stroke_data import preprocess_data
from scripts.train_and_evaluate_models import evaluate_models

def main():
    print("=== Stroke Prediction Analysis ===")
    
    # Step 1: Preprocess the dataset
    input_path = "data/raw/healthcare-dataset-stroke-data.csv"
    output_path = "data/processed/healthcare_data_preprocessed.csv"
    print("\n[Step 1] Running data preprocessing...")
    preprocess_data(input_path, output_path)
    print("[Step 1] Preprocessing completed successfully!")

    # Step 2: Train and evaluate models
    print("\n[Step 2] Running model training and evaluation...")
    evaluate_models(output_path)
    print("[Step 2] Training and evaluation completed successfully!")

    print("\n=== Analysis Completed Successfully ===")
    print("Results have been generated. Please check the outputs.")

if __name__ == "__main__":
    main()
