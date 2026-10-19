import torch
from math import sqrt

def flash_attention_forward(Q, K, V, block_size=2):
    seq_len, d_model = Q.shape
    scale = 1.0 / sqrt(d_model)
    O = torch.zeros_like(Q)

    for i in range(0, seq_len, block_size):
        Q_i = Q[i:i + block_size]
        m_i = torch.full((Q_i.shape[0],), -float('inf'),
                         device=Q.device, dtype=Q.dtype)
        l_i = torch.zeros(Q_i.shape[0], device=Q.device, dtype=Q.dtype)
        O_i = torch.zeros_like(Q_i)

        for j in range(0, seq_len, block_size):
            K_j, V_j = K[j:j + block_size], V[j:j + block_size]
            S_ij = (Q_i @ K_j.T) * scale
            m_ij = S_ij.max(dim=1).values            # 注意 .values
            m_new = torch.maximum(m_i, m_ij)

            scale_old = torch.exp(m_i - m_new)
            P_ij = torch.exp(S_ij - m_new[:, None])

            l_i = scale_old * l_i + P_ij.sum(dim=1)
            O_i = scale_old[:, None] * O_i + P_ij @ V_j
            m_i = m_new

        O[i:i + block_size] = O_i / l_i[:, None]

    return O