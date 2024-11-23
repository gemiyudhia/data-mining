import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def preprocess_data(input_path: str, output_path: str):
    """
    Preprocesses the stroke dataset.
    - Handles missing values
    - Encodes categorical features
    - Scales numerical features
    - Saves the preprocessed data to a CSV file.

    Args:
        input_path (str): Path to the raw dataset.
        output_path (str): Path to save the preprocessed dataset.
    """
    # Load the dataset
    try:
        print(f"Loading data from {input_path}...")
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"[ERROR] File not found at {input_path}")
        return

    print("Dataset loaded successfully.")

    # Handle missing values in BMI
    print("Handling missing values...")
    imputer = SimpleImputer(strategy='mean')
    df['bmi'] = imputer.fit_transform(df[['bmi']])

    # One-hot encode categorical variables
    print("Encoding categorical variables...")
    categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    # Scale numerical variables
    print("Scaling numerical variables...")
    numerical_columns = ['age', 'avg_glucose_level', 'bmi']
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    # Save preprocessed data
    print(f"Saving preprocessed data to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Preprocessing completed and data saved.")

if __name__ == "__main__":
    # Example usage
    preprocess_data("../data/raw/healthcare-dataset-stroke-data.csv", "../data/processed/healthcare_data_preprocessed.csv")
