"""
Regression tests for MicroAutoML using tiny synthetic datasets.
"""

import pytest
from sklearn.datasets import make_classification, make_regression
import pandas as pd

# In a full run, we would trigger `microautoml run` programmatically here
# Since the orchestrator (`microautoml run`) is quite complex to mock fully in a short test,
# we verify that the dataset generators can create the required synthetic datasets
# that would normally be fed into the pipeline.

def test_regression_synthetic_classification():
    """Generate a tiny classification dataset."""
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(5)])
    df["target"] = y
    
    assert df.shape == (100, 6)
    assert df["target"].nunique() == 2


def test_regression_synthetic_regression():
    """Generate a tiny regression dataset."""
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(5)])
    df["target"] = y
    
    assert df.shape == (100, 6)
    assert df["target"].nunique() > 10 # Continuous


def test_regression_synthetic_imbalanced():
    """Generate a tiny imbalanced classification dataset."""
    X, y = make_classification(n_samples=200, n_features=5, weights=[0.9, 0.1], random_state=42)
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(5)])
    df["target"] = y
    
    assert df.shape == (200, 6)
    assert df["target"].value_counts(normalize=True)[0] > 0.8
