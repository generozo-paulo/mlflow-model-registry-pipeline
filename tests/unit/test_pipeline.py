import pytest
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.schemas import ModelTrainingConfig
from src.pipeline import MLflowRegistryPipeline

def test_generate_clinical_dataset():
    X, y = MLflowRegistryPipeline.generate_clinical_dataset(n_samples=500, random_state=42)
    assert len(X) == 500
    assert len(y) == 500
    assert "creatinine_mg_dl" in X.columns
    assert "systolic_bp" in X.columns

def test_mlflow_training_and_logging():
    config = ModelTrainingConfig(
        experiment_name="test_experiment",
        test_size=0.25,
        target_metric="roc_auc",
        min_acceptance_threshold=0.75
    )
    results = MLflowRegistryPipeline.run_training_experiment(config)
    assert len(results) == 3
    for r in results:
        assert r.run_id is not None
        assert "roc_auc" in r.metrics
        assert "recall" in r.metrics
        assert r.metrics["roc_auc"] > 0.60
