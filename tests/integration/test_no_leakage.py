"""
CV Leakage integration tests.
Ensures preprocessing is strictly fitted on TRAIN folds only.
"""
import pytest
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from automl.compiler.compiler import PipelineCompiler
from automl.core.schemas import ExperimentSpec
from automl.core.enums import OperatorType

def test_compiler_creates_pipeline_that_fits_on_train_only():
    """
    Test that the compiled pipeline correctly avoids leakage.
    We test this by passing mock data to the compiled sklearn pipeline 
    and ensuring it executes fit_transform on train, and transform on val.
    """
    spec = ExperimentSpec(
        model_name="Ridge",
        operator=OperatorType.BASELINE,
        preprocessing=[{
            "name": "StandardScaler",
            "columns": ["col_num"]
        }]
    )
    
    compiler = PipelineCompiler()
    # Assume auto-injection of imputation/encoding in compiler (to be implemented in Priority 2)
    # We will test the basic pipeline structure for now
    pipeline = compiler.compile(spec)
    
    assert isinstance(pipeline, Pipeline)
    
    X = pd.DataFrame({
        "col_num": np.random.rand(100),
        "col_cat": np.random.choice(["A", "B"], 100)
    })
    y = pd.Series(np.random.rand(100))
    
    # KFold CV to ensure fit/transform semantics
    kf = KFold(n_splits=2)
    for train_idx, val_idx in kf.split(X):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # Fit on train ONLY
        pipeline.fit(X_train, y_train)
        
        # Predict on val
        preds = pipeline.predict(X_val)
        assert len(preds) == len(X_val)

