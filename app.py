import plotly.express as px
from src.analytics_ml import detect_transaction_anomalies, forecast_monthly_revenue
from src.data_engine import generate_enterprise_dataset, load_uploaded_csv
from src.nlq_engine import execute_analytical_query
from src.privacy_guard import apply_privacy_pipeline
import streamlit as st

st.set_page_config(
    page_title="BI Guard | Governança & Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilo limpo para cards e cabeçalhos
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.8rem; padding-bottom: 2rem; }
    div[data-testid="stMetric"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 14px 18px;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# SIDEBAR: FONTE DE DADOS E GOVERNANÇA
# ==========================================
with st.sidebar:
    st.header("Configurações do Pipeline")

    data_source = st.radio(
        "Fonte de Dados",
        ["Base Corporativa (Sintética)", "Upload CSV Externo"],
        horizontal=False,
    )

    if data_source == "Upload CSV Externo":
        uploaded = st.file_uploader("Selecione um arquivo CSV", type=["csv"])
        raw_df = (
            load_uploaded_csv(uploaded)
            if uploaded
            else generate_enterprise_dataset()
        )
    else:
        raw_df = generate_enterprise_dataset()

    st.divider()
    st.subheader("Governança e Privacidade (LGPD)")
    lgpd_enabled = st.toggle("Camada de Proteção LGPD", value=True)

    dp_mode = st.checkbox(
        "Privacidade Diferencial (Laplace)",
        value=False,
        disabled=not lgpd_enabled,
        help="Adiciona ruído estatístico calibrado pelo parâmetro Epsilon (ε).",
    )
    epsilon_val = None
    if lgpd_enabled and dp_mode:
        epsilon_val = st.slider(
            "Orçamento de Privacidade (ε)",
            min_value=0.1,
            max_value=2.0,
            value=0.8,
            step=0.1,
        )

    st.divider()
    st.subheader("Filtros Analíticos")

    if "region" in raw_df.columns:
        selected_regions = st.multiselect(
            "Regiões",
            options=sorted(raw_df["region"].unique()),
            default=sorted(raw_df["region"].unique()),
        )
        raw_df = raw_df[raw_df["region"].isin(selected_regions)]

    if "date" in raw_df.columns and not raw_df.empty:
        min_d, max_d = raw_df["date"].min().date(), raw_df["date"].max().date()
        date_range = st.date_input(
            "Período", value=(min_d, max_d), min_value=min_d, max_value=max_d
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            raw_df = raw_df[
                (raw_df["date"].dt.date >= date_range[0])
                & (raw_df["date"].dt.date <= date_range[1])
            ]

# Aplica governança antes de qualquer renderização
df_clean, audit_metadata = apply_privacy_pipeline(
    raw_df, enabled=lgpd_enabled, epsilon=epsilon_val
)

# ==========================================
# CABEÇALHO PRINCIPAL
# ==========================================
col_title, col_status = st.columns([3, 1])
with col_title:
    st.title("BI Guard — Analytics & Governança")
    st.caption(
        "Arquitetura de Business Intelligence com Machine Learning, NLQ e"
        " Conformidade LGPD"
    )

with col_status:
    if lgpd_enabled:
        st.success("Status: Conformidade LGPD Ativa")
    else:
        st.error("Status: Dados PII Expostos")

if df_clean.empty:
    st.warning("Nenhum registro encontrado para os filtros selecionados.")
    st.stop()

# ==========================================
# NAVEGAÇÃO POR ABAS
# ==========================================
tab_overview, tab_ml, tab_nlq, tab_audit = st.tabs([
    "Visão Executiva",
    "Modelos Preditivos & Anomalias",
    "Consulta em Linguagem Natural (NLQ)",
    "Auditoria LGPD & Linhagem",
])

# ------------------------------------------
# ABA 1: VISÃO EXECUTIVA
# ------------------------------------------
with tab_overview:
    k1, k2, k3, k4 = st.columns(4)
    total_rev = df_clean["revenue"].sum()
    avg_ticket = df_clean["revenue"].mean()
    tx_count = len(df_clean)
    avg_churn = df_clean["churn_probability"].mean() * 100

    k1.metric("Receita Acumulada", f"R$ {total_rev:,.2f}")
    k2.metric("Ticket Médio", f"R$ {avg_ticket:,.2f}")
    k3.metric("Transações Processadas", f"{tx_count:,}".replace(",", "."))
    k4.metric("Risco Médio de Churn", f"{avg_churn:.1f}%")

    st.markdown("---")
    c1, c2 = st.columns([3, 2])

    with c1:
        st.subheader("Desempenho por Região e Linha de Produto")
        grouped = df_clean.groupby(["region", "category"], as_index=False)[
            "revenue"
        ].sum()
        fig_bar = px.bar(
            grouped,
            x="region",
            y="revenue",
            color="category",
            barmode="group",
            labels={
                "region": "Região",
                "revenue": "Receita (R$)",
                "category": "Categoria",
            },
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig_bar.update_layout(
            margin=dict(t=20, b=20, l=10, r=10), legend_title_text=""
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with c2:
        st.subheader("Participação por Categoria")
        cat_share = df_clean.groupby("category", as_index=False)[
            "revenue"
        ].sum()
        fig_pie = px.pie(
            cat_share,
            names="category",
            values="revenue",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig_pie.update_layout(margin=dict(t=20, b=20, l=10, r=10))
        st.plotly_chart(fig_pie, use_container_width=True)

# ------------------------------------------
# ABA 2: MODELOS PREDITIVOS & ANOMALIAS
# ------------------------------------------
with tab_ml:
    col_f1, col_f2 = st.columns([3, 1])
    with col_f1:
        st.subheader("Projeção de Receita (Regressão Ridge com Sazonalidade)")
    with col_f2:
        horizon = st.selectbox(
            "Horizonte de Previsão (meses)", [3, 6, 9], index=0
        )

    forecast_df = forecast_monthly_revenue(df_clean, periods_ahead=horizon)
    fig_forecast = px.line(
        forecast_df,
        x="month",
        y="revenue",
        color="type",
        markers=True,
        labels={"month": "Competência", "revenue": "Receita (R$)", "type": ""},
        color_discrete_map={
            "Histórico": "#1f77b4",
            "Projeção (Ridge ML)": "#d62728",
        },
    )
    fig_forecast.update_layout(margin=dict(t=15, b=20))
    st.plotly_chart(fig_forecast, use_container_width=True)

    st.subheader("Detecção Não-Supervisionada de Anomalias (Isolation Forest)")
    anomalies_df = detect_transaction_anomalies(df_clean, contamination=0.03)
    st.caption(
        f"Foram isoladas {len(anomalies_df)} transações com comportamento"
        " estatístico atípico entre receita e risco de churn."
    )
    if not anomalies_df.empty:
        cols_show = [
            c
            for c in [
                "transaction_id",
                "subject_token",
                "date",
                "region",
                "category",
                "revenue",
                "churn_probability",
            ]
            if c in anomalies_df.columns
        ]
        st.dataframe(
            anomalies_df[cols_show].head(10),
            use_container_width=True,
            hide_index=True,
        )

# ------------------------------------------
# ABA 3: CONSULTA EM LINGUAGEM NATURAL (NLQ)
# ------------------------------------------
with tab_nlq:
    st.subheader("Motor Analítico Conversacional")
    st.markdown(
        "Consulte métricas e dimensões em português. A camada de governança"
        " remove atributos identificadores antes do processamento."
    )

    presets = [
        "Qual o faturamento total por região?",
        "Qual o ticket médio por categoria?",
        "Qual o risco médio de churn por região?",
        "Qual o volume de transações por mês?",
    ]
    selected_preset = st.pills(
        "Consultas rápidas:", presets, selection_mode="single"
    )
    user_query = st.text_input(
        "Pergunta analítica:",
        value=selected_preset if selected_preset else presets[0],
    )

    if user_query:
        response = execute_analytical_query(
            df=df_clean,
            question=user_query,
            pii_cols=audit_metadata["detected_pii_columns"],
            lgpd_active=lgpd_enabled,
        )

        st.info(response["summary"])

        c_res1, c_res2 = st.columns([2, 3])
        with c_res1:
            st.dataframe(
                response["table"], use_container_width=True, hide_index=True
            )
        with c_res2:
            fig_nlq = px.bar(
                response["table"],
                x=response["x_col"],
                y=response["y_col"],
                color_discrete_sequence=["#2b5c8f"],
            )
            fig_nlq.update_layout(margin=dict(t=10, b=10))
            st.plotly_chart(fig_nlq, use_container_width=True)

        with st.expander("Inspecionar Metadados de Privacidade da Consulta"):
            st.json(response["governance_payload"])

# ------------------------------------------
# ABA 4: AUDITORIA LGPD & LINHAGEM
# ------------------------------------------
with tab_audit:
    st.subheader("Registro de Auditoria e Amostra de Dados Processados")
    col_a1, col_a2 = st.columns([1, 2])

    with col_a1:
        st.markdown("**Metadados de Execução**")
        st.json(audit_metadata)

    with col_a2:
        st.markdown(
            "**Visualização da Camada de Dados (25 primeiros registros)**"
        )
        st.dataframe(
            df_clean.head(25), use_container_width=True, hide_index=True
        )
        
