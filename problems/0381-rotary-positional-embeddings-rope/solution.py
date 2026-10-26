import numpy as np

def apply_rope(x: np.ndarray, positions: np.ndarray, base: float = 10000.0) -> np.ndarray:
    """
    Apply Rotary Positional Embeddings (RoPE) to input embeddings.
    
    Args:
        x: Input embeddings of shape (seq_len, d), d must be even
        positions: Position indices of shape (seq_len,)
        base: Base for frequency computation (default: 10000.0)
    
    Returns:
        Embeddings with rotary positional encoding applied, shape (seq_len, d)
    """
    seq_len, d = x.shape[0], x.shape[1]
    x_1, x_2 = x[:, ::2], x[:, 1::2]
    half = d // 2
    theta = base ** (-np.arange(half) * 2.0 / d) 
    pt = np.outer(positions, theta)  # seq_len, d // 2

    y_1, y_2 = np.cos(pt) * x_1 - np.sin(pt) * x_2, np.sin(pt) * x_1 + np.cos(pt) * x_2

    out = np.empty_like(x)
    out[:, ::2] = y_1
    out[:, 1::2] = y_2
    return out

    pass