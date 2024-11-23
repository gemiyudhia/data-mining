import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

def load_and_preprocess_data(data_path):
    # Load dataset
    df = pd.read_csv(data_path)

    # Handle missing values
    numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
    numerical_imputer = SimpleImputer(strategy='mean')
    df[numerical_features] = numerical_imputer.fit_transform(df[numerical_features])

    categorical_features = df.select_dtypes(include=['object']).columns
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_features] = categorical_imputer.fit_transform(df[categorical_features])

    # Encode categorical variables
    label_encoder = LabelEncoder()
    for feature in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']:
        df[feature] = label_encoder.fit_transform(df[feature])

    # Split features and target
    X = df.drop(columns=['id', 'stroke'])
    y = df['stroke']

    # Normalize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Oversample minority class
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    return X_resampled, y_resampled, X.columns
