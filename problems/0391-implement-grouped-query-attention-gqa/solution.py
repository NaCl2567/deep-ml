import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x)        
    y = np.exp(x)
    return y / np.sum(y, axis=axis, keepdims=True)


def grouped_query_attention(Q, K, V, num_heads, num_kv_heads):
    """
    Compute Grouped Query Attention.
    
    Args:
        Q: Query tensor, shape (batch_size, seq_len, num_heads * head_dim)
        K: Key tensor, shape (batch_size, seq_len, num_kv_heads * head_dim)
        V: Value tensor, shape (batch_size, seq_len, num_kv_heads * head_dim)
        num_heads: Number of query heads
        num_kv_heads: Number of key/value heads
    
    Returns:
        Output tensor, shape (batch_size, seq_len, num_heads * head_dim)
    """
    batch_size, seq_len, head_dim = Q.shape[0], Q.shape[1], Q.shape[2] // num_heads
    Q = Q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    K = K.reshape(batch_size, seq_len, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
    V = V.reshape(batch_size, seq_len, num_kv_heads, head_dim).transpose(0, 2, 1, 3)

    group_num = num_heads // num_kv_heads
    K = np.repeat(K, group_num, axis=1)
    V = np.repeat(V, group_num, axis=1)

    # return np.matmul(Q, K.transpose(0,1,3,2)).shape
    # return  V.shape

    A = np.matmul(softmax(np.matmul(Q, K.transpose(0,1,3,2)) / np.sqrt(head_dim), axis=-1), V)
    # return A.shape
    O = A.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, num_heads * head_dim)
    
    return O