import torch
from math import sqrt
import numpy as np
def compute_qkv(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor):
    """
    Compute Query (Q), Key (K), and Value (V) matrices.
    """
    return torch.matmul(X, W_q), torch.matmul(X, W_k), torch.matmul(X, W_v)


def masked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray) -> np.ndarray:
	"""
	Compute masked self-attention.
	"""

	A = torch.matmul(Q, K.T) / sqrt(Q.shape[-1])

	if mask is not None:
		A = A + mask

    A = torch.softmax(A, dim=-1)
    return torch.matmul(A, V)