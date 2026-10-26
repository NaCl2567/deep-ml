import numpy as np

def softmax(x):
    x = np.asarray(x, dtype=float)                 # don't mutate input, accept lists
    x = x - np.max(x, axis=-1, keepdims=True)      # keepdims -> correct broadcast
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)

def kv_cache_attention_step(x_new: np.ndarray, W_Q: np.ndarray, W_K: np.ndarray, W_V: np.ndarray, cache: tuple) -> tuple:
    """
    Perform a single attention step with KV caching.
    
    Args:
        x_new: New token embedding, shape (d_model,)
        W_Q: Query projection matrix, shape (d_model, d_k)
        W_K: Key projection matrix, shape (d_model, d_k)
        W_V: Value projection matrix, shape (d_model, d_v)
        cache: Tuple (K_cache, V_cache) or None if first step
    
    Returns:
        Tuple (output, updated_cache)
    """
    d_k = W_K.shape[1]
    q_new, k_new, v_new = x_new @ W_Q, x_new @ W_K, x_new @ W_V

    if cache:
        k_cache, v_cache = cache
        k = np.concatenate([k_cache, k_new[None, :]], axis=0)   # seq_len, d_k
        v = np.concatenate([v_cache, v_new[None, :]], axis=0)
    else:
        k = k_new[None, :]
        v = v_new[None, :]

    output = softmax(q_new @ k.T / np.sqrt(d_k)) @ v  # d_v

    # k_add, v_add = output @ W_K, output @ W_V

    # return ((None), None, k_add.shape)

    # k, v = np.concatenate([k, k_add[None, :]], axis=0), np.concatenate([v, v_add[None, :]], axis=0)

    return (output, (k, v))
