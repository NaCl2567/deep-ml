import numpy as np
from scipy.special import softmax
from math import sqrt

def paged_attention(
    query: np.ndarray,
    key_cache: np.ndarray,
    value_cache: np.ndarray,
    block_table: list,
    context_len: int
) -> np.ndarray:
    """
    Perform scaled dot-product attention with paged KV cache.

    Args:
        query:       (num_heads, head_dim) query for a single token
        key_cache:   (num_physical_blocks, block_size, num_heads, head_dim)
        value_cache: (num_physical_blocks, block_size, num_heads, head_dim)
        block_table: list of physical block indices (logical -> physical)
        context_len: number of valid KV tokens in the sequence

    Returns:
        (num_heads, head_dim) attention output, rounded to 4 decimal places
    """
    num_blocks, block_size, num_heads, head_dim = key_cache.shape
    # print(num_blocks, block_size)
    logical_index = list(range((context_len + block_size - 1) // block_size))
    # assert len(logical_index) == 2, "not ok"
    physical_index = np.array(block_table)[logical_index]

    assert key_cache[physical_index].shape == (len(physical_index), block_size, num_heads, head_dim), "not ok"

    K = key_cache[physical_index].reshape(-1, num_heads, head_dim)[:context_len].transpose(1, 0, 2)    # num_heads, context_len, head_dim
    V = value_cache[physical_index].reshape(-1, num_heads, head_dim)[:context_len].transpose(1, 0, 2) # num_heads, context_len, head_dim
    # print(physical_index)


    scores = (query[:, None, :] @ K.transpose(0, 2, 1)) / sqrt(head_dim)  # num_heads, 1, context_len
    # scores -= np.max(scores, axis=None)
    alpha = softmax(scores, axis=-1)
    
    assert alpha.shape == (num_heads, 1, context_len), "message"
    o = alpha @ V   # num_heads, 1, head_dim,
    

    return o[:, 0, :]





    pass