import numpy as np
import math

# ========================================
# KL Divergence Tree
# ========================================

def kl_divergence(p, q):
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    mask = (p > 0) & (q > 0)
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))

def kl_impurity(y_left, y_right):
    n_total = len(y_left) + len(y_right)
    classes = np.unique(np.concatenate([y_left, y_right]))
    
    def class_dist(y):
        counts = np.array([np.sum(y == c) for c in classes], dtype=float)
        if counts.sum() == 0:
            return np.ones_like(counts) / len(counts)
        return counts / counts.sum()
    
    p_parent = class_dist(np.concatenate([y_left, y_right]))
    p_left = class_dist(y_left)
    p_right = class_dist(y_right)
    
    return (len(y_left) / n_total) * kl_divergence(p_left, p_parent) + \
           (len(y_right) / n_total) * kl_divergence(p_right, p_parent)

class KLDecisionTree:
    def __init__(self, max_depth=4, min_samples_split=10, n_random_directions=20, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_random_directions = n_random_directions
        self.random_state = np.random.RandomState(random_state)
        self.tree_ = None

    def best_split(self, X, y):
        best_score = -np.inf
        best_params = None
        n_samples, n_features = X.shape
        for _ in range(self.n_random_directions):
            f = self.random_state.randint(0, n_features)
            thresholds = np.unique(X[:, f])
            for t in thresholds:
                left_mask = X[:, f] <= t
                right_mask = X[:, f] > t
                if len(y[left_mask]) < self.min_samples_split or len(y[right_mask]) < self.min_samples_split:
                    continue
                score = kl_impurity(y[left_mask], y[right_mask])
                if score > best_score:
                    best_score = score
                    best_params = (f, t)
        return best_params

    def build_tree(self, X, y, depth=0):
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return np.bincount(y, minlength=np.max(y)+1).argmax()
        split = self.best_split(X, y)
        if split is None:
            return np.bincount(y, minlength=np.max(y)+1).argmax()
        f, t = split
        left_mask = X[:, f] <= t
        right_mask = X[:, f] > t
        left_child = self.build_tree(X[left_mask], y[left_mask], depth+1)
        right_child = self.build_tree(X[right_mask], y[right_mask], depth+1)
        return {'feature': f, 'threshold': t, 'left': left_child, 'right': right_child}

    def fit(self, X, y):
        self.tree_ = self.build_tree(X, y)
        return self

    def _predict_one(self, x, node):
        if not isinstance(node, dict):
            return node
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        else:
            return self._predict_one(x, node['right'])

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree_) for x in X])

# ========================================
# Tsallis Divergence Tree
# ========================================

def tsallis_impurity(y, q=1.5):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    if q == 1:
        return -np.sum(probs * np.log2(probs))
    else:
        return (1 - np.sum(probs ** q)) / (q - 1)

class TsallisDecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2, q=1.5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.q = q
        self.tree_ = None

    def best_split(self, X, y):
        best_score = -np.inf
        best_split = None
        n_features = X.shape[1]
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = X[:, feature] > threshold
                if len(y[left_mask]) < self.min_samples_split or len(y[right_mask]) < self.min_samples_split:
                    continue
                parent_impurity = tsallis_impurity(y, q=self.q)
                left_impurity = tsallis_impurity(y[left_mask], q=self.q)
                right_impurity = tsallis_impurity(y[right_mask], q=self.q)
                n = len(y)
                score = parent_impurity - (len(y[left_mask]) / n) * left_impurity - (len(y[right_mask]) / n) * right_impurity
                if score > best_score:
                    best_score = score
                    best_split = (feature, threshold)
        return best_split

    def build_tree(self, X, y, depth=0):
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return np.bincount(y, minlength=np.max(y)+1).argmax()
        split = self.best_split(X, y)
        if split is None:
            return np.bincount(y, minlength=np.max(y)+1).argmax()
        feature, threshold = split
        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold
        left_child = self.build_tree(X[left_mask], y[left_mask], depth+1)
        right_child = self.build_tree(X[right_mask], y[right_mask], depth+1)
        return {"feature": feature, "threshold": threshold, "left": left_child, "right": right_child}

    def fit(self, X, y):
        self.tree_ = self.build_tree(X, y)
        return self

    def _predict_one(self, x, node):
        if not isinstance(node, dict):
            return node
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_one(x, node["left"])
        else:
            return self._predict_one(x, node["right"])

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree_) for x in X])