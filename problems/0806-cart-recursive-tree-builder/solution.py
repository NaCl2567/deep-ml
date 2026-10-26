import numpy as np

def gini_impurity(y):
    if len(y) == 0:
        return 0
    k = max(y) + 1
    N = len(y)
    result = 1
    for i in range(k):
        result -= (sum(y == i) / N) * (sum(y == i) / N)
    return result

def gini_gain_feature(X, y, feature):
    N, k = X.shape[0], X.shape[1]
    origin_impurity = gini_impurity(y)

    feature_col = X[:, feature]
    sample_and_feature = [(i, feature_col[i]) for i in range(N)]
    sample_and_feature = sorted(sample_and_feature, key=lambda x: x[1])

    left_samples = []
    right_samples = [x[0] for x in sample_and_feature]

    max_gain = 0
    best_threshold = None
    for i in range(len(sample_and_feature) - 1):
        left_samples += [sample_and_feature[i][0]]
        right_samples = right_samples[1:]

        split_impurity = len(left_samples) / N * gini_impurity(y[left_samples]) + len(right_samples) / N * gini_impurity(y[right_samples])

        gain = origin_impurity - split_impurity
        if gain > max_gain:
            best_threshold = (sample_and_feature[i][1] + sample_and_feature[i+1][1]) / 2
            max_gain = gain
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
    if best_feature_threshold is None or max_gain == 0:
        return {'leaf': True, 'prediction': int(np.bincount(y.astype(int)).argmax())}

    feature, threshold = best_feature_threshold

    mask = X[:, feature] <= threshold

    if mask.all() or (~mask).all():
        return TreeNode(leaf=True,
                        prediction=np.bincount(y.astype(int)).argmax())

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