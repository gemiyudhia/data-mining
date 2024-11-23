import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(y_true, y_pred, model_name):
    """
    Evaluates a model and prints the confusion matrix and performance metrics.
    """
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"--- {model_name} ---")
    print("Confusion Matrix:\n", cm)
    print(f"Accuracy: {acc:.2f}")
    print(f"Precision: {prec:.2f}")
    print(f"Recall: {rec:.2f}")
    print(f"F1-Score: {f1:.2f}")
    print("\nClassification Report:\n", classification_report(y_true, y_pred))
    return cm, acc, prec, rec, f1

def evaluate_models(data_path):
    """
    Loads preprocessed data, trains models, evaluates them, and visualizes the results.
    """
    # Load preprocessed data
    df = pd.read_csv(data_path)

    # Split features and target
    X = df.drop(columns=['id', 'stroke'])
    y = df['stroke']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize models
    naive_bayes = GaussianNB()
    knn = KNeighborsClassifier(n_neighbors=5)

    # Train models
    naive_bayes.fit(X_train, y_train)
    knn.fit(X_train, y_train)

    # Predictions
    y_pred_nb = naive_bayes.predict(X_test)
    y_pred_knn = knn.predict(X_test)

    # Evaluate models
    print("Evaluating Naïve Bayes")
    evaluate_model(y_test, y_pred_nb, "Naïve Bayes")

    print("Evaluating K-Nearest Neighbor")
    evaluate_model(y_test, y_pred_knn, "K-Nearest Neighbor")

    # Visualize confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(confusion_matrix(y_test, y_pred_nb), annot=True, fmt="d", cmap="Blues", ax=axes[0])
    axes[0].set_title("Naïve Bayes Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    sns.heatmap(confusion_matrix(y_test, y_pred_knn), annot=True, fmt="d", cmap="Greens", ax=axes[1])
    axes[1].set_title("KNN Confusion Matrix")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.tight_layout()
    plt.show()
