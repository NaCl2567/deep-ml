import numpy as np

def gini_impurity(y):
    if len(y) == 0:
        return 0
    k = int(max(y) + 1)
    N = len(y)
    result = 1
    for i in range(k):
        result -= (sum(y == i) / N) * (sum(y == i) / N)
    return result

def gini_gain_feature(X, y, feature):
    N = X.shape[0]
    origin = gini_impurity(y)
    order = np.argsort(X[:, feature], kind='mergesort')   # 稳定排序
    vals  = X[order, feature]

    max_gain, best_threshold = 0.0, None
    for split in range(1, N):
        if vals[split - 1] == vals[split]:                # 相同值不能切
            continue
        left_idx,  right_idx = order[:split], order[split:]
        n_l, n_r = split, N - split
        gain = (origin
                - (n_l / N) * gini_impurity(y[left_idx])
                - (n_r / N) * gini_impurity(y[right_idx]))
        if gain > max_gain:
            max_gain = gain
            best_threshold = (vals[split - 1] + vals[split]) / 2
    return max_gain, best_threshold

def gini_gain(X, y):
    N, k = X.shape[0], X.shape[1]
    max_gain = 0
    best_feature_threshold = None
    for feature in range(k):
        cur_gain, cur_threshold = gini_gain_feature(X, y, feature)
        if cur_gain > max_gain:
            max_gain = cur_gain
            best_feature_threshold = (feature, cur_threshold)
    
    return max_gain, best_feature_threshold


def build_cart_tree(X, y, max_depth=3, min_samples_split=2, depth=0):
    """
    Build a CART classification tree using Gini impurity.

    Returns a nested dict representing the tree.
    """

    # X : num_samples, num_features
    # y : num_samples
    if X.shape[0] < min_samples_split or depth >= max_depth:
        return {'leaf': True, 'prediction': int(np.bincount(y.astype(int)).argmax())}

    max_gain, best_feature_threshold = gini_gain(X, y)
    if best_feature_threshold is None or max_gain <= 1e-12:
        return {'leaf': True, 'prediction': int(np.bincount(y.astype(int)).argmax())}

    feature, threshold = best_feature_threshold

    mask = X[:, feature] <= threshold

    if mask.all() or (~mask).all():
        return {
            'leaf': True,        
            'prediction': int(np.bincount(y.astype(int)).argmax())
            }

    left_X,  left_y  = X[mask],  y[mask]
    right_X, right_y = X[~mask], y[~mask]
    root = {
        'leaf': False, 
        'feature': int(feature), 
        'threshold': float(threshold), 
        'left': build_cart_tree(
            left_X, 
            left_y, 
            max_depth=max_depth, 
            min_samples_split=min_samples_split,
            depth=depth+1
            ), 
        'right': build_cart_tree(
            right_X, 
            right_y, 
            max_depth=max_depth, 
            min_samples_split=min_samples_split,
            depth=depth+1
            )
        }

    return root