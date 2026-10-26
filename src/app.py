import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.schemas import ModelTrainingConfig
from src.pipeline import MLflowRegistryPipeline

st.set_page_config(
    page_title="MLflow Model Registry & Tracking Pipeline",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 MLflow Model Registry & Enterprise Governance Pipeline")
st.markdown("""
Plataforma industrial de **MLOps** para rastreamento de experimentos, versionamento de artefatos, governança clínica e promoção de modelos através de **Quality Gates** automáticos.
""")

tabs = st.tabs(["🚀 1. Executar Experimento & Rastreamento", "🏛️ 2. Model Registry & Quality Gates", "📋 3. Linhagem e Governança Regulatória"])

# -------------------------------------------------------------
# TAB 1: Execução
# -------------------------------------------------------------
with tabs[0]:
    st.header("Configuração do Experimento de Machine Learning")
    col1, col2 = st.columns(2)
    
    with col1:
        exp_name = st.text_input("Nome do Experimento MLflow", "clinical_mortality_registry")
        target_metric = st.selectbox("Métrica Primária de Avaliação", ["roc_auc", "recall", "f1"], index=0)
        threshold = st.slider("Quality Gate Mínimo Aceitável para Produção", 0.70, 0.95, 0.82, 0.01)
        test_size = st.slider("Fração do Conjunto de Teste", 0.15, 0.40, 0.25, 0.05)
        
        btn_run = st.button("▶️ Disparar Treinamento & Log no MLflow", type="primary")
        
    with col2:
        st.subheader("Painel de Resultados dos Modelos Concorrentes")
        
        config = ModelTrainingConfig(
            experiment_name=exp_name,
            test_size=test_size,
            target_metric=target_metric,
            min_acceptance_threshold=threshold
        )
        
        if btn_run or "last_results" not in st.session_state:
            with st.spinner("Treinando modelos e registrando artefatos no MLflow..."):
                results = MLflowRegistryPipeline.run_training_experiment(config)
                st.session_state["last_results"] = results
        else:
            results = st.session_state["last_results"]
            
        summary_rows = []
        for r in results:
            summary_rows.append({
                "Modelo": r.model_type,
                "ROC-AUC": r.metrics["roc_auc"],
                "Recall (Sensibilidade)": r.metrics["recall"],
                "F1-Score": r.metrics["f1"],
                "Acurácia": r.metrics["accuracy"],
                "Quality Gate": "APROVADO ✅" if r.passed_quality_gate else "REPROVADO ❌",
                "Run ID": r.run_id[:8] + "..."
            })
            
        df_summary = pd.DataFrame(summary_rows)
        st.dataframe(df_summary.style.format({"ROC-AUC": "{:.4f}", "Recall (Sensibilidade)": "{:.4f}", "F1-Score": "{:.4f}", "Acurácia": "{:.4f}"}))
        
        fig = px.bar(
            df_summary,
            x="Modelo",
            y=["ROC-AUC", "Recall (Sensibilidade)", "F1-Score"],
            barmode="group",
            title="Benchmark Comparativo Multimétricas"
        )
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# TAB 2: Model Registry
# -------------------------------------------------------------
with tabs[1]:
    st.header("Gerenciamento de Estágios (Model Registry)")
    st.write("""
    Modelos aprovados pelo Quality Gate são elegíveis para transição de estágio:
    * **None $\\rightarrow$ Staging:** Disponível para validação de equipe médica / testes de regressão.
    * **Staging $\\rightarrow$ Production:** Ativo para inferência em tempo real com monitoramento de latência e drift.
    * **Production $\\rightarrow$ Archived:** Versões anteriores aposentadas mantidas para auditoria histórica.
    """)
    
    st.info(f"**Backend MLflow Ativo:** SQLite local persistente (`mlflow.db`) com rastreamento completo de parâmetros e pesos.")

# -------------------------------------------------------------
# TAB 3: Linhagem
# -------------------------------------------------------------
with tabs[2]:
    st.header("Linhagem de Dados & Conformidade Regulatória")
    st.markdown("""
    * **Reprodutibilidade Estrita:** Cada Run armazena o hash dos dados de entrada, semente aleatória e versão das bibliotecas.
    * **Conformidade em Saúde:** Registros imutáveis em conformidade com exigências de validação clínica e governança da informação.
    """)
