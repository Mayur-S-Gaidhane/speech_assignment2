# ===============================
# LID Evaluation Metrics
# ===============================

from sklearn.metrics import f1_score, confusion_matrix

def evaluate(y_true, y_pred):

    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)

    print("F1 Score:", f1)
    print("Confusion Matrix:\n", cm)

    return f1, cm