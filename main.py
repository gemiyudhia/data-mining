from scripts.preprocessing import load_and_preprocess_data
from scripts.training import train_models
from scripts.evaluation import evaluate_model, visualize_results
from sklearn.model_selection import train_test_split

# File path
data_path = "data/raw/healthcare-dataset-stroke-data.csv"
processed_path = "data/processed/healthcare_data_preprocessed.csv"

# Preprocess data
X_resampled, y_resampled, feature_columns = load_and_preprocess_data(data_path)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train models
naive_bayes, knn = train_models(X_train, y_train)

# Predictions
y_pred_nb = naive_bayes.predict(X_test)
y_pred_knn = knn.predict(X_test)

# Evaluate models
cm_nb, acc_nb, prec_nb, rec_nb, f1_nb = evaluate_model(y_test, y_pred_nb, "Naïve Bayes")
cm_knn, acc_knn, prec_knn, rec_knn, f1_knn = evaluate_model(y_test, y_pred_knn, "KNN")

# Visualize results
metrics = {
    "Accuracy": [acc_nb, acc_knn],
    "Precision": [prec_nb, prec_knn],
    "Recall": [rec_nb, rec_knn],
    "F1-Score": [f1_nb, f1_knn]
}
visualize_results(cm_nb, cm_knn, metrics)
