# Spec-Driven Development (SDD) Specification Sheet
## Project: `mlflow-model-registry-pipeline`

* **Version:** 1.0.0
* **Author:** Paulo Elias Tavares Generozo
* **Domain:** MLOps, Model Governance, Lifecycle Management & Experiment Tracking
* **License:** MIT

---

## 1. Business & MLOps Problem Statement (CRISP-DM Phase 1)

In production enterprise environments, machine learning projects frequently degrade or fail because:
1. **Lack of Experiment Tracking:** Data scientists tune models locally without logging hyperparameters, dataset hashes, or evaluation artifacts.
2. **Uncontrolled Model Deployment:** Models are moved to production via manual file copying (`model.pkl`) without automated validation or rollback mechanisms.
3. **Regulatory Auditability:** In healthcare and finance, regulatory bodies require complete lineage (*Who trained this model? On what exact data? With which performance metrics?*).

This project implements an **end-to-end MLOps Pipeline** using **MLflow Tracking and Model Registry**, executing multi-model hyperparameter tuning (`RandomForest`, `XGBoost`, `LogisticRegression`), evaluating against strict production quality gates, and managing transitions across stages (`Staging` $\rightarrow$ `Production`).

---

## 2. Data Contracts & Schemas (Pydantic Models)

### Training Configuration Schema (`ModelTrainingConfig`):
```python
class ModelTrainingConfig:
    experiment_name: str = "clinical_mortality_prediction"
    test_size: float = 0.25
    random_state: int = 42
    target_metric: str = "roc_auc" # "roc_auc" | "recall" | "f1"
    min_acceptance_threshold: float = 0.85
```

### Model Registry Status Schema (`ModelRegistryStatus`):
```python
class ModelRegistryStatus:
    model_name: str
    latest_version: int
    current_stage: str # "None" | "Staging" | "Production" | "Archived"
    run_id: str
    metrics: dict[str, float]
    promoted_to_production: bool
```

---

## 3. Performance & Quality Targets (SLAs)

* **Experiment Lineage:** 100% of runs log parameters, metrics, and serialized artifacts.
* **Quality Gate Enforcement:** No model can transition to `Production` unless its `roc_auc` $\ge 0.85$ and `recall` $\ge 0.80$.
* **Unit Test Coverage:** $\ge 90\%$ test coverage on training, evaluation, and registry logic.
