from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import os

os.makedirs("outputs", exist_ok=True)

def evaluate_model(name, model, X_test, y_test):
    print(f"\n🔍 Evaluating: {name}")
    
    # Make predictions
    y_pred = model.predict(X_test)

    # Print standard metrics
    print("Accuracy     :", accuracy_score(y_test, y_pred))
    print("Precision    :", precision_score(y_test, y_pred))
    print("Recall       :", recall_score(y_test, y_pred))
    print("F1-Score     :", f1_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{name} - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"outputs/confusion_matrix_{name.replace(' ', '_').lower()}.png")
    plt.show()