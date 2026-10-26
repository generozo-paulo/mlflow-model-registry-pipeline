# CRISP-DM: MLOps & Business Understanding
## Projeto: `mlflow-model-registry-pipeline`

### 1. Desafio de Engenharia e Governança
Em aplicações de inteligência artificial corporativa, o maior índice de falhas decorre do chamado "abismo de deploy":
* Modelos treinados localmente perdem reprodutibilidade com o tempo (*environment drift*).
* Falta de versionamento formal de hiperparâmetros impede investigações de falhas preditivas em ambiente de produção.
* Ausência de portões de qualidade automatizados (*Quality Gates*) permite que modelos com baixa sensibilidade (falsos negativos críticos) sejam promovidos acidentalmente.

### 2. Solução com MLflow
Este projeto estabelece uma infraestrutura de governança padronizada:
* Rastreamento centralizado de parâmetros, métricas e artefatos binários (`mlflow.sklearn`).
* Aplicação de Quality Gates baseados em métricas clínicas essenciais (**ROC-AUC $\ge 0.82$** e **Recall**).
* Segregação e controle de ciclo de vida em estágios formais (*Staging*, *Production*, *Archived*).
