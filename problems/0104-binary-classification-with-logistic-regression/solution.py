import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def predict_logistic(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    z = X @ weights + bias
    p = sigmoid(z)
    return np.where(p >= 0.5, 1, 0)