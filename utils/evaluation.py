from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import os

def evaluate(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=class_names,
                yticklabels=class_names)

    os.makedirs("outputs/plots", exist_ok=True)
    plt.savefig("outputs/plots/confusion_matrix.png")

    report = classification_report(y_true, y_pred)

    with open("outputs/plots/classification_report.txt", "w") as f:
        f.write(report)

    print(report)