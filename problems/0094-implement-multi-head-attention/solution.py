import torch
import torch.nn.functional as F
from typing import Tuple
from math import sqrt
def compute_qkv(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Query, Key, and Value matrices.

    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)

    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    # Your code here
    W_qkv = torch.cat([W_q, W_k, W_v], dim=-1) 
    QKV = torch.matmul(X, W_qkv)
    Q, K, V = QKV.chunk(3, dim=-1)
    return Q, K, V

def self_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product self-attention.

    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)

    Returns:
        Attention output of shape (seq_len, d_k)
    """
    # Your code here
    return torch.matmul(torch.softmax(torch.matmul(Q, K.transpose(-1, -2)) / sqrt(Q.shape[-1]), dim=-1), V)

def multi_head_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, n_heads: int) -> torch.Tensor:
    """
    Compute multi-head attention.

    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads

    Returns:
        Attention output of shape (seq_len, d_model)
    """
    d_model = Q.shape[-1]
    if d_model % n_heads:
        print("error")
        return None
    head_dim = d_model // n_heads
    Q, K, V = Q.reshape(-1, n_heads, head_dim).transpose(0, 1), K.reshape(-1, n_heads, head_dim).transpose(0, 1), V.reshape(-1, n_heads, head_dim).transpose(0, 1)    # seq_len * n_heads, head_dim
    att = self_attention(Q, K, V)

    return att.transpose(0, 1).reshape(-1, d_model)
