# ===============================
# Equal Error Rate (EER)
# ===============================

import numpy as np
from sklearn.metrics import roc_curve

def calculate_eer(y_true, y_scores):
    """
    Compute Equal Error Rate (EER)

    y_true: 0 (spoof), 1 (real)
    y_scores: prediction scores
    """

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    fnr = 1 - tpr

    # Find point where FPR ≈ FNR
    eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]

    print("EER:", eer)

    return eer