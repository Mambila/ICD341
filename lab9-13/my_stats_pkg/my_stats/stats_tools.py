import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

def compute_median(values):
    """Compute the median of a non-empty sequence of numbers."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("Input cannot be empty.")
    return float(np.median(arr))

def estimate_probability(successes, trials):
    """Estimate a Bernoulli probability."""
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("Invalid successes or trials count.")
    return successes / trials

class StdScalerTransformer(BaseEstimator, TransformerMixin):
    """Scales each column of a 2D array by its standard deviation."""
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.std_ = np.std(X, axis=0)
        # Avoid division by zero for constant columns
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return X / self.std_