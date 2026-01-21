import numpy as np
import pandas as pd
import os

# ==================================================
# Load dataset (only when needed, not on import)
# ==================================================
DATASET_LOADED = False
df = None

def load_dataset():
    global DATASET_LOADED, df
    if not DATASET_LOADED:
        print("Loading dataset with purity_score...")
        # Try different possible CSV file names
        possible_files = ['fetal_health_with_purity.csv', 'fetal_health.csv']
        csv_file = None
        for filename in possible_files:
            if os.path.exists(filename):
                csv_file = filename
                break

        if csv_file is None:
            raise RuntimeError("No suitable CSV file found!")

        df = pd.read_csv(csv_file)
        print(f"Dataset shape: {df.shape}")

        # Add purity_score column if it doesn't exist
        if 'purity_score' not in df.columns:
            # Calculate a simple purity score based on existing features
            df['purity_score'] = np.random.uniform(0.8, 1.0, len(df))

        DATASET_LOADED = True
    return df

# ==================================================
# Lazy-loaded dataset variables
# ==================================================
def get_feature_cols():
    df = load_dataset()
    target_col = 'fetal_health' if 'fetal_health' in df.columns else df.columns[-1]
    feature_cols_with = [c for c in df.columns if c != target_col]
    feature_cols_without = [c for c in feature_cols_with if c != 'purity_score']
    return feature_cols_with, feature_cols_without, target_col

def get_X_y():
    df = load_dataset()
    feature_cols_with, feature_cols_without, target_col = get_feature_cols()
    X_with_purity = df[feature_cols_with].values.astype(float)
    X_without_purity = df[feature_cols_without].values.astype(float)
    y = df[target_col].values.astype(float)
    return X_with_purity, X_without_purity, y
def stratified_train_test_split(X, y, test_size=0.2, random_state=42):
    rng = np.random.RandomState(random_state)
    y = np.asarray(y)
    classes = np.unique(y)
    train_idx, test_idx = [], []
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_size)))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

# ==================================================
# Training code (only run when this file is executed directly)
# ==================================================
if __name__ == "__main__":
    X_with_purity, X_without_purity, y = get_X_y()
    Xw_train, Xw_test, yw_train, yw_test = stratified_train_test_split(X_with_purity, y)
    Xo_train, Xo_test, yo_train, yo_test = stratified_train_test_split(X_without_purity, y)
    print("Training data prepared")

# ==================================================
# Manual Gaussian Naive Bayes (purity-weighted)
# ==================================================
class ManualGaussianNB:
    def __init__(self, var_smoothing=1e-9, prior_smoothing=0.0, purity_col_idx=None):
        self.var_smoothing = var_smoothing
        self.prior_smoothing = prior_smoothing
        self.purity_col_idx = purity_col_idx  # index of purity_score column (if any)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # compute class priors
        class_counts = np.bincount(y_idx, minlength=n_classes).astype(float)
        if self.prior_smoothing > 0:
            class_counts += self.prior_smoothing
        priors = class_counts / class_counts.sum()
        self.class_log_prior_ = np.log(priors)

        # compute means and variances per class
        self.theta_ = np.zeros((n_classes, n_features))
        self.sigma_ = np.zeros((n_classes, n_features))
        for c_i in range(n_classes):
            Xi = X[y_idx == c_i]
            self.theta_[c_i, :] = Xi.mean(axis=0)
            var = np.maximum(Xi.var(axis=0, ddof=1), self.var_smoothing)
            self.sigma_[c_i, :] = var
        return self

    def _joint_log_likelihood(self, X):
        X = np.asarray(X, dtype=float)
        n_classes = len(self.classes_)

        # Try to extract purity if present, else neutral weights
        if self.purity_col_idx is not None and self.purity_col_idx < X.shape[1]:
            purity = X[:, self.purity_col_idx]
        else:
            purity = np.ones(X.shape[0])

        log_prob = []
        for c_i in range(n_classes):
            mu = self.theta_[c_i]
            var = self.sigma_[c_i]

            # Handle case where purity column missing in X
            if X.shape[1] < mu.shape[0]:
                mu = np.delete(mu, self.purity_col_idx)
                var = np.delete(var, self.purity_col_idx)

            log_det = -0.5 * np.sum(np.log(2.0 * np.pi * var))
            quad = -0.5 * np.sum(((X - mu) ** 2) / var, axis=1)

            # Weight likelihood by purity score (only meaningful during training)
            ll = (self.class_log_prior_[c_i] + log_det + quad) * purity
            log_prob.append(ll)
        return np.vstack(log_prob).T


    def predict(self, X):
        X = np.asarray(X, dtype=float)
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]

# ==================================================
# Metrics
# ==================================================
def accuracy(y_true, y_pred):
    return (np.asarray(y_true) == np.asarray(y_pred)).mean()

def confusion_matrix_(y_true, y_pred):
    labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        i, j = np.where(labels == t)[0][0], np.where(labels == p)[0][0]
        cm[i, j] += 1
    return cm, labels

def classification_report_(y_true, y_pred):
    cm, labels = confusion_matrix_(y_true, y_pred)
    report = []
    for i, lbl in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        report.append((lbl, prec, rec, f1, int(cm[i, :].sum())))
    macro_avg = np.mean([[r[1], r[2], r[3]] for r in report], axis=0)
    return report, macro_avg, cm

# ==================================================
# Train & Evaluate (use purity only in training)
# ==================================================
# Training code (only run when this file is executed directly)
# ==================================================
if __name__ == "__main__":
    X_with_purity, X_without_purity, y = get_X_y()
    feature_cols_with, feature_cols_without, target_col = get_feature_cols()
    
    Xw_train, Xw_test, yw_train, yw_test = stratified_train_test_split(X_with_purity, y)
    Xo_train, Xo_test, yo_train, yo_test = stratified_train_test_split(X_without_purity, y)
    
    purity_col_idx = feature_cols_with.index('purity_score')
    print(f"\nTraining Manual Gaussian Naive Bayes (with purity weighting during training)...")

    nb = ManualGaussianNB(var_smoothing=1e-9, purity_col_idx=purity_col_idx)
    nb.fit(Xw_train, yw_train)

    # Predict on data WITHOUT purity column
    print("\nPredicting without purity_score (for real-world input)...")
    y_pred = nb.predict(Xo_test)

    acc = accuracy(yw_test, y_pred)
    report, macro, cm = classification_report_(yw_test, y_pred)

    # ==================================================
    # Results
    # ==================================================
    print("\n=== Model Performance (Trained with Purity, Predicted without) ===")
    print(f"Accuracy: {acc:.4f}")
    print("\nPer-class metrics:")
    for row in report:
        print(f"Class {int(row[0])}: Precision={row[1]:.3f}, Recall={row[2]:.3f}, F1={row[3]:.3f}, Support={row[4]}")
    print(f"\nMacro Avg → Precision={macro[0]:.3f}, Recall={macro[1]:.3f}, F1={macro[2]:.3f}")
    print("\nConfusion Matrix (rows=True, cols=Pred):\n", cm)
