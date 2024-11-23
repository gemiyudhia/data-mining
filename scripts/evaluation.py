import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
import pandas as pd

def evaluate_model(y_true, y_pred, model_name):
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

def visualize_results(cm_nb, cm_knn, metrics):
    # Visualize confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cm_nb, annot=True, fmt="d", cmap="Blues", ax=axes[0])
    axes[0].set_title("Naïve Bayes Confusion Matrix")
    sns.heatmap(cm_knn, annot=True, fmt="d", cmap="Greens", ax=axes[1])
    axes[1].set_title("KNN Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # Compare metrics
    metrics_df = pd.DataFrame(metrics, index=["Naïve Bayes", "KNN"])
    metrics_df.plot(kind="bar", figsize=(10, 6))
    plt.title("Comparison of Model Performance Metrics")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.show()
