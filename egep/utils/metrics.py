import numpy as np
import math

def calculate_metrics(y_pred, y_true):
    y_pred = np.asarray(y_pred).flatten()
    y_true = np.asarray(y_true).flatten()
    
    # RMSE
    mse = np.mean((y_pred - y_true) ** 2)
    rmse = math.sqrt(mse)
    
    # MAPE (Avoid division by zero)
    raw_ape = np.abs((y_pred - y_true) / (y_true + 1e-5))
    mask = np.isfinite(raw_ape)
    mape = np.mean(raw_ape[mask]) if mask.any() else 0.0
    
    # RAE
    numerator = np.sum(np.abs(y_pred - y_true))
    denominator = np.sum(np.abs(y_true - np.mean(y_true)))
    rae = numerator / (denominator + 1e-5)
    
    return {"RMSE": rmse, "MAPE": mape, "RAE": rae}