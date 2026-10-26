import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, recall_score, f1_score, accuracy_score
from typing import List, Dict, Any, Tuple
from src.schemas import ModelTrainingConfig, ModelRunResult

class MLflowRegistryPipeline:
    """
    Pipeline MLOps para Rastreamento de Experimentos, Versionamento e Model Registry com MLflow.
    """

    @staticmethod
    def setup_tracking(experiment_name: str = "clinical_mortality_registry") -> str:
        # Habilita FileStore no MLflow local
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        mlruns_dir = os.path.abspath("mlruns").replace("\\", "/")
        tracking_uri = f"file:///{mlruns_dir}"
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        return tracking_uri

    @staticmethod
    def generate_clinical_dataset(n_samples: int = 2000, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
        """Gera dataset sintético com variáveis clínicas (biomarcadores, pressão, idade)."""
        X, y = make_classification(
            n_samples=n_samples,
            n_features=12,
            n_informative=8,
            n_classes=2,
            weights=[0.75, 0.25], # Desbalanceamento clínico realista
            random_state=random_state
        )
        feature_names = [
            "age", "systolic_bp", "creatinine_mg_dl", "albumin_g_dl",
            "glucose_mg_dl", "hemoglobin_g_dl", "crp_mg_l", "heart_rate_bpm",
            "oxygen_sat_pct", "bun_mg_dl", "sodium_mmol_l", "potassium_mmol_l"
        ]
        return pd.DataFrame(X, columns=feature_names), pd.Series(y, name="mortality_30d")

    @staticmethod
    def run_training_experiment(config: ModelTrainingConfig) -> List[ModelRunResult]:
        MLflowRegistryPipeline.setup_tracking(config.experiment_name)
        X, y = MLflowRegistryPipeline.generate_clinical_dataset(random_state=config.random_state)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.test_size, random_state=config.random_state, stratify=y
        )

        candidates = [
            ("RandomForest_HighRecall", RandomForestClassifier(n_estimators=100, max_depth=6, class_weight='balanced', random_state=config.random_state)),
            ("GradientBoosting_Tuned", GradientBoostingClassifier(n_estimators=120, learning_rate=0.08, max_depth=4, random_state=config.random_state)),
            ("LogisticRegression_L2", LogisticRegression(max_iter=500, class_weight='balanced', random_state=config.random_state))
        ]

        results = []

        for name, model in candidates:
            with mlflow.start_run(run_name=name) as run:
                # Log de parâmetros
                params = {k: str(v) for k, v in model.get_params().items() if k in ['n_estimators', 'max_depth', 'learning_rate', 'class_weight', 'max_iter']}
                mlflow.log_params(params)

                # Treinamento
                model.fit(X_train, y_train)

                # Inferência e Métricas
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]

                metrics = {
                    "roc_auc": float(roc_auc_score(y_test, y_proba)),
                    "recall": float(recall_score(y_test, y_pred)),
                    "f1": float(f1_score(y_test, y_pred)),
                    "accuracy": float(accuracy_score(y_test, y_pred))
                }
                mlflow.log_metrics(metrics)

                # Log de artefato do modelo com cloudpickle para evitar bloqueios de auditoria skops
                mlflow.sklearn.log_model(
                    model, 
                    artifact_path="model",
                    serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
                )

                # Quality Gate
                score = metrics.get(config.target_metric, 0.0)
                passed = score >= config.min_acceptance_threshold
                status_msg = f"Aprovado para Staging ({config.target_metric}={score:.4f})" if passed else f"Reprovado pelo Quality Gate ({score:.4f} < {config.min_acceptance_threshold})"

                results.append(ModelRunResult(
                    run_id=run.info.run_id,
                    model_type=name,
                    params=params,
                    metrics=metrics,
                    passed_quality_gate=passed,
                    status_message=status_msg
                ))

        return results
