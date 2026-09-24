"""
Test boundary guard invariant tests.
"""
import pytest
import pandas as pd

from automl.core.schemas import EvidencePack, SplitProfile
from automl.core.enums import SplitType
from automl.application.test_guard import TestBoundaryGuard
from automl.application.invariants import InvariantViolation

def test_test_rows_never_reach_runner():
    # Setup data
    X = pd.DataFrame({"col1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    # Setup evidence with a split
    split = SplitProfile(split_type=SplitType.IID_CV, test_indices=[8, 9], train_val_indices=[0, 1, 2, 3, 4, 5, 6, 7])
    evidence = EvidencePack(split=split)

    # Slice for search
    X_train_val, y_train_val = TestBoundaryGuard.slice_for_search(X, y, evidence)
    
    assert len(X_train_val) == 8
    assert 8 not in X_train_val.index
    assert 9 not in X_train_val.index

def test_test_set_used_exactly_once():
    split = SplitProfile(split_type=SplitType.IID_CV, test_indices=[8, 9], train_val_indices=[0, 1, 2, 3, 4, 5, 6, 7])
    
    assert not split.test_set_used
    split.mark_test_used()
    assert split.test_set_used
    
    with pytest.raises(Exception):
        split.mark_test_used()

def test_test_guard_corrupted_split_raises():
    X = pd.DataFrame({"col1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    
    # Overlapping split
    split = SplitProfile(split_type=SplitType.IID_CV, test_indices=[7, 8, 9], train_val_indices=[0, 1, 2, 3, 4, 5, 6, 7])
    evidence = EvidencePack(split=split)
    
    with pytest.raises(InvariantViolation):
        TestBoundaryGuard.slice_for_search(X, y, evidence)
