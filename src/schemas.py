from pydantic import BaseModel, Field
from typing import Dict, Literal, Optional

class ModelTrainingConfig(BaseModel):
    experiment_name: str = Field(default="clinical_mortality_registry")
    test_size: float = Field(default=0.25, ge=0.1, le=0.5)
    random_state: int = Field(default=42)
    target_metric: Literal["roc_auc", "recall", "f1"] = "roc_auc"
    min_acceptance_threshold: float = Field(default=0.82, ge=0.5, le=0.99)

class ModelRunResult(BaseModel):
    run_id: str
    model_type: str
    params: Dict[str, str]
    metrics: Dict[str, float]
    passed_quality_gate: bool
    status_message: str
