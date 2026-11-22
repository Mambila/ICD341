import numpy as np
import pytest
from my_stats.stats_tools import compute_median, estimate_probability, StdScalerTransformer

# (1) Median tests: known result and empty input
def test_median_known_result():
    assert compute_median([3, 1, 2]) == 2.0
    assert compute_median([1, 2, 3, 4]) == 2.5

def test_median_empty_input():
    with pytest.raises(ValueError):
        compute_median([])

# (2) Probability estimate lies in [0, 1]
@pytest.mark.parametrize("successes,trials", [(0, 10), (5, 10), (10, 10), (8, 12)])
def test_probability_in_unit_interval(successes, trials):
    p = estimate_probability(successes, trials)
    assert 0.0 <= p <= 1.0

# (3) Transformer scales each column by its std
def test_std_scaler_transformer():
    X = np.array([[2.0, 10.0], [4.0, 20.0], [6.0, 30.0]])
    transformer = StdScalerTransformer()
    X_scaled = transformer.fit_transform(X)
    assert np.allclose(np.std(X_scaled, axis=0), [1.0, 1.0])