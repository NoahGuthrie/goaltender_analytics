import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from pathlib import Path
import json

def evaluate_model(y_true, y_prob, model_name="Model", output_dir="data/models/eval"):
    """
    Computes Log Loss, Brier Score, and AUC-ROC.
    Saves metrics to JSON and plots a calibration curve.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Metrics
    ll = log_loss(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.5 # If only one class is present in a tiny test set
        
    metrics = {
        "log_loss": ll,
        "brier_score": brier,
        "auc": auc
    }
    
    with open(Path(output_dir) / f"{model_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"--- {model_name} Evaluation ---")
    print(f"Log Loss: {ll:.4f}")
    print(f"Brier:    {brier:.4f}")
    print(f"AUC-ROC:  {auc:.4f}")
    
    # 2. Calibration Curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.plot(prob_pred, prob_true, "s-", label=f"{model_name}")
    plt.ylabel("Fraction of positives")
    plt.xlabel("Mean predicted probability")
    plt.title(f"Calibration Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(Path(output_dir) / f"{model_name}_calibration.png")
    plt.close()
    
    return metrics
