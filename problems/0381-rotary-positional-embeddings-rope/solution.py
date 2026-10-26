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
    # if x.shape[0] <= 1:
    #     return x
    # Your code here
    seq_len, d = x.shape[0], x.shape[1]
    x_1, x_2 = x[:, ::2], x[:, 1::2]
    theta = np.array([1 / np.power(base, 2 * i / d) for i in range(d // 2)])
    # positions_1, positions_2 = np.array(positions[::2]), np.array(positions[1::2])

    pt = positions[:, None] @ theta[None, :]    # seq_len, d // 2

    x_1, x_2 = np.cos(pt) * x_1 - np.sin(pt) * x_2, np.sin(pt) * x_1 + np.cos(pt) * x_2

    return np.concatenate([np.concatenate([x_1[:, i:i+1], x_2[:, i:i+1]], axis=1) for i in range(d//2)], axis=1)


    pass