import hashlib
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import streamlit as st

# ==========================================
# CONFIGURAÇÃO DA PÁGINA
# ==========================================
st.set_page_config(
    page_title="BI Guard & AI Insights | UNISO", page_icon="📊", layout="wide"
)


# ==========================================
# GERAÇÃO DE DATASET SINTÉTICO (BIG DATA / VAREJO)
# ==========================================
@st.cache_data
def carregar_dados(n_registros=500):
    np.random.seed(42)
    datas = pd.date_range(start="2025-01-01", end="2026-10-31", freq="D")
    regioes = ["Sudeste", "Sul", "Nordeste", "Centro-Oeste", "Norte"]
    categorias = [
        "Eletrônicos",
        "Cloud & Software",
        "Acessórios",
        "Consultoria TI",
    ]
    nomes = [
        "Ana Silva",
        "Carlos Souza",
        "Mariana Lima",
        "João Pereira",
        "Fernanda Costa",
        "Lucas Oliveira",
        "Beatriz Santos",
        "Rafael Almeida",
        "Juliana Rocha",
        "Pedro Martins",
    ]

    df = pd.DataFrame({
        "ID_Transacao": [f"TRX-{10000+i}" for i in range(n_registros)],
        "Data": np.random.choice(datas, n_registros),
        "Cliente_Nome": np.random.choice(nomes, n_registros),
        "CPF": [
            (
                f"{np.random.randint(100,999)}.{np.random.randint(100,999)}."
                f"{np.random.randint(100,999)}-{np.random.randint(10,99)}"
            )
            for _ in range(n_registros)
        ],
        "Email": [
            f"cliente{i}@exemplo.com.br"
            for i in np.random.randint(1, 250, n_registros)
        ],
        "Regiao": np.random.choice(
            regioes, n_registros, p=[0.45, 0.20, 0.18, 0.10, 0.07]
        ),
        "Categoria": np.random.choice(categorias, n_registros),
        "Valor_Venda": np.round(np.random.uniform(150.0, 4800.0, n_registros), 2),
        "Score_Risco_Churn": np.round(np.random.uniform(0.05, 0.95, n_registros), 2),
    })
    df["Data"] = pd.to_datetime(df["Data"])
    df["Mes_Ano"] = df["Data"].dt.to_period("M").astype(str)
    return df.sort_values("Data").reset_index(drop=True)


# ==========================================
# FUNÇÕES DE PRIVACIDADE E GOVERNANÇA (LGPD)
# ==========================================
def aplicar_governanca_lgpd(df, modo_lgpd, nivel_privacidade):
    df_proc = df.copy()
    if not modo_lgpd:
        return df_proc

    # Pseudonimização (Hash SHA-256 truncado)
    df_proc["ID_Cliente_Hash"] = df_proc["CPF"].apply(
        lambda x: hashlib.sha256(x.encode()).hexdigest()[:12].upper()
    )

    # Mascaramento de PII (Personally Identifiable Information)
    df_proc["Cliente_Nome"] = df_proc["Cliente_Nome"].apply(
        lambda x: x[0] + "*** " + x.split()[-1][0] + "***"
    )
    df_proc["CPF"] = df_proc["CPF"].apply(lambda x: "***.***." + x[8:])
    df_proc["Email"] = df_proc["Email"].apply(
        lambda x: x[:2] + "****@" + x.split("@")[1]
    )

    # Privacidade Diferencial (Adição de ruído estatístico controlado nos valores)
    if nivel_privacidade == "Alta (Anonimização + Privacidade Diferencial)":
        ruido = np.random.laplace(0, 50, size=len(df_proc))
        df_proc["Valor_Venda"] = np.maximum(
            0, np.round(df_proc["Valor_Venda"] + ruido, 2)
        )

    return df_proc


# ==========================================
# BARRA LATERAL (CONTROLES DE GOVERNANÇA)
# ==========================================
st.sidebar.title("🛡️ Governança & Privacidade")
st.sidebar.markdown(
    "Controle de conformidade **LGPD** aplicado em tempo real sobre o"
    " pipeline de BI."
)

modo_lgpd = st.sidebar.toggle("Ativar Proteção de Dados (LGPD)", value=True)
nivel_privacidade = st.sidebar.radio(
    "Nível de Proteção:",
    [
        "Padrão (Mascaramento + Pseudonimização)",
        "Alta (Anonimização + Privacidade Diferencial)",
    ],
    disabled=not modo_lgpd,
)

st.sidebar.divider()
st.sidebar.subheader("Filtros Analíticos")
df_bruto = carregar_dados(600)

regioes_selecionadas = st.sidebar.multiselect(
    "Filtrar Regiões:",
    options=df_bruto["Regiao"].unique(),
    default=df_bruto["Regiao"].unique(),
)

df_filtrado = df_bruto[df_bruto["Regiao"].isin(regioes_selecionadas)]
df_exibicao = aplicar_governanca_lgpd(df_filtrado, modo_lgpd, nivel_privacidade)

# ==========================================
# CABEÇALHO PRINCIPAL
# ==========================================
st.title("📊 BI Guard & AI Insights")
st.caption(
    "Demonstração Prática: Tendências Futuras, IA Generativa e Privacidade em"
    " Big Data"
)

if modo_lgpd:
    st.success(
        "🔒 **Modo LGPD Ativo:** Dados pessoais sensíveis (PII) mascarados"
        " antes da visualização e bloqueados para envio a modelos de IA."
    )
else:
    st.warning(
        "⚠️ **Modo LGPD Desativado:** Dados sensíveis (Nome, CPF, E-mail)"
        " expostos em texto claro. Risco de não conformidade!"
    )

# ==========================================
# ABAS DA APLICAÇÃO
# ==========================================
aba1, aba2, aba3 = st.tabs([
    "📈 1. BI Descritivo & Preditivo",
    "🤖 2. Assistente BI com IA (NLQ)",
    "🔐 3. Auditoria de Privacidade & Dados",
])

# ------------------------------------------
# ABA 1: BI DESCRITIVO + PREDITIVO (ML)
# ------------------------------------------
with aba1:
    col1, col2, col3, col4 = st.columns(4)
    receita_total = df_exibicao["Valor_Venda"].sum()
    ticket_medio = df_exibicao["Valor_Venda"].mean()
    total_vendas = len(df_exibicao)
    risco_medio = df_exibicao["Score_Risco_Churn"].mean() * 100

    col1.metric("Receita Total", f"R$ {receita_total:,.2f}")
    col2.metric("Ticket Médio", f"R$ {ticket_medio:,.2f}")
    col3.metric("Volume de Transações", f"{total_vendas}")
    col4.metric("Risco Médio de Churn", f"{risco_medio:.1f}%")

    st.divider()
    c_graf1, c_graf2 = st.columns(2)

    with c_graf1:
        st.subheader("Receita por Região e Categoria")
        fig_bar = px.bar(
            df_exibicao.groupby(["Regiao", "Categoria"], as_index=False)[
                "Valor_Venda"
            ].sum(),
            x="Regiao",
            y="Valor_Venda",
            color="Categoria",
            barmode="group",
            template="plotly_white",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with c_graf2:
        st.subheader("Tendência Mensal + Previsão (Machine Learning)")
        mensal = (
            df_exibicao.groupby("Mes_Ano", as_index=False)["Valor_Venda"]
            .sum()
            .sort_values("Mes_Ano")
        )
        mensal["Indice"] = np.arange(len(mensal))

        # Modelo simples de Regressão Linear para projeção futura (3 meses)
        modelo = LinearRegression()
        modelo.fit(mensal[["Indice"]], mensal["Valor_Venda"])

        futuros_idx = np.arange(len(mensal), len(mensal) + 3)
        previsoes = modelo.predict(futuros_idx.reshape(-1, 1))

        df_prev = pd.DataFrame({
            "Mes_Ano": ["2026-11 (Prev)", "2026-12 (Prev)", "2027-01 (Prev)"],
            "Valor_Venda": previsoes,
            "Tipo": "Projeção IA",
        })
        mensal["Tipo"] = "Histórico"
        df_tendencia = pd.concat(
            [mensal[["Mes_Ano", "Valor_Venda", "Tipo"]], df_prev],
            ignore_index=True,
        )

        fig_line = px.line(
            df_tendencia,
            x="Mes_Ano",
            y="Valor_Venda",
            color="Tipo",
            markers=True,
            template="plotly_white",
        )
        st.plotly_chart(fig_line, use_container_width=True)

# ------------------------------------------
# ABA 2: ASSISTENTE DE BI COM IA (NLQ)
# ------------------------------------------
with aba2:
    st.subheader("Consulta em Linguagem Natural (Natural Language Query)")
    st.markdown(
        "Faça perguntas sobre os dados. O motor de IA analisa apenas metadados"
        " agregados (sem acesso a PII)."
    )

    perguntas_prontas = [
        "Selecione uma pergunta exemplo ou digite abaixo...",
        "Qual região possui o maior faturamento acumulado?",
        "Qual categoria tem o maior risco médio de churn?",
        "Qual é o resumo executivo e recomendação estratégica?",
    ]
    escolha = st.selectbox("Sugestões rápidas:", perguntas_prontas)
    pergunta_usuario = st.text_input(
        "Ou digite sua pergunta:",
        value="" if escolha == perguntas_prontas[0] else escolha,
    )

    if st.button("Gerar Insight com IA", type="primary"):
        if not pergunta_usuario.strip():
            st.info("Digite ou selecione uma pergunta para analisar.")
        else:
            p = pergunta_usuario.lower()
            st.markdown("### 🧠 Resposta do Copiloto Analítico:")

            # Mostra o payload seguro enviado à IA
            with st.expander(
                "Ver Payload JSON enviado ao Modelo de IA (Camada de"
                " Privacidade)"
            ):
                colunas_permitidas = [
                    "Regiao",
                    "Categoria",
                    "Valor_Venda",
                    "Score_Risco_Churn",
                ]
                st.json({
                    "lgpd_filter_applied": modo_lgpd,
                    "blocked_columns": (
                        ["Cliente_Nome", "CPF", "Email"]
                        if modo_lgpd
                        else ["NENHUMA (RISCO!)"]
                    ),
                    "aggregated_sample": (
                        df_exibicao[colunas_permitidas]
                        .head(3)
                        .to_dict(orient="records")
                    ),
                })

            if "região" in p or "regiao" in p or "faturamento" in p:
                top_reg = (
                    df_exibicao.groupby("Regiao")["Valor_Venda"]
                    .sum()
                    .sort_values(ascending=False)
                )
                lider = top_reg.index[0]
                valor_lider = top_reg.iloc[0]
                st.write(
                    f"A região **{lider}** lidera o faturamento com **R$"
                    f" {valor_lider:,.2f}**, representando"
                    f" **{(valor_lider/receita_total)*100:.1f}%** da receita"
                    " total filtrada."
                )
                st.bar_chart(top_reg)

            elif "churn" in p or "risco" in p or "categoria" in p:
                risco_cat = (
                    df_exibicao.groupby("Categoria")["Score_Risco_Churn"]
                    .mean()
                    .sort_values(ascending=False)
                )
                cat_critica = risco_cat.index[0]
                st.write(
                    f"A categoria **{cat_critica}** apresenta o maior índice"
                    f" médio de risco de cancelamento (Churn) em **{risco_cat.iloc[0]*100:.1f}%**. "
                    "Recomenda-se ação preventiva de retenção para clientes"
                    " deste segmento."
                )
                st.bar_chart(risco_cat)

            else:
                top_cat = (
                    df_exibicao.groupby("Categoria")["Valor_Venda"]
                    .sum()
                    .idxmax()
                )
                top_reg = (
                    df_exibicao.groupby("Regiao")["Valor_Venda"].sum().idxmax()
                )
                st.write(f"""
                **Diagnóstico Automatizado:**
                * **Motor de Receita:** A combinação de maior impacto concentra-se na região **{top_reg}** e na vertical **{top_cat}**.
                * **Alerta de Governança:** O pipeline está operando com proteção LGPD **{'ATIVADA' if modo_lgpd else 'DESATIVADA'}**.
                * **Próximo Passo Sugerido:** Priorizar campanhas de upsell na região {top_reg} e mitigar churn nos contratos de alto ticket.
                """)

# ------------------------------------------
# ABA 3: AUDITORIA DE PRIVACIDADE & DADOS
# ------------------------------------------
with aba3:
    st.subheader("Inspeção da Tabela de Dados (Antes vs. Depois da LGPD)")
    st.markdown(
        "Alterne o botão **'Ativar Proteção de Dados (LGPD)'** na barra"
        " lateral para observar o mascaramento, pseudonimização (Hash) e"
        " ruído de Privacidade Diferencial."
    )

    colunas_mostrar = [
        "ID_Transacao",
        "Data",
        "Cliente_Nome",
        "CPF",
        "Email",
        "Regiao",
        "Categoria",
        "Valor_Venda",
    ]
    if modo_lgpd and "ID_Cliente_Hash" in df_exibicao.columns:
        colunas_mostrar.insert(2, "ID_Cliente_Hash")

    st.dataframe(df_exibicao[colunas_mostrar].head(20), use_container_width=True)