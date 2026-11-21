import os
import pandas as pd


def execute_analytical_query(
    df: pd.DataFrame, question: str, pii_cols: list[str], lgpd_active: bool
) -> dict:
    """
    Executa consultas analíticas dinâmicas (NLQ) garantindo que colunas PII nunca entrem no contexto.
    """
    safe_cols = [c for c in df.columns if c not in pii_cols]
    safe_df = df[safe_cols].copy()

    q = question.lower().strip()

    # Identifica dimensão de agrupamento
    if any(w in q for w in ["região", "regiao", "estado", "local"]):
        dim = "region"
        dim_label = "Região"
    elif any(w in q for w in ["categoria", "produto", "serviço", "servico"]):
        dim = "category"
        dim_label = "Categoria"
    elif any(w in q for w in ["mês", "mes", "tempo", "evolução", "evolucao"]):
        safe_df["period"] = safe_df["date"].dt.to_period("M").astype(str)
        dim = "period"
        dim_label = "Período"
    else:
        dim = "category"
        dim_label = "Categoria"

    # Identifica métrica e agregação
    if any(w in q for w in ["churn", "risco", "cancelamento"]):
        metric = "churn_probability"
        agg_func = "mean"
        metric_label = "Risco Médio de Churn (%)"
        multiplier = 100
    elif any(w in q for w in ["ticket", "média", "media", "médio", "medio"]):
        metric = "revenue"
        agg_func = "mean"
        metric_label = "Ticket Médio (R$)"
        multiplier = 1
    elif any(
        w in q for w in ["quantidade", "volume", "contagem", "transações"]
    ):
        metric = "revenue"
        agg_func = "count"
        metric_label = "Volume de Transações"
        multiplier = 1
    else:
        metric = "revenue"
        agg_func = "sum"
        metric_label = "Receita Total (R$)"
        multiplier = 1

    result_df = (
        safe_df.groupby(dim, as_index=False)[metric]
        .agg(agg_func)
        .rename(columns={dim: dim_label, metric: metric_label})
    )
    result_df[metric_label] = (result_df[metric_label] * multiplier).round(2)
    result_df = result_df.sort_values(
        by=metric_label, ascending=False
    ).reset_index(drop=True)

    top_row = result_df.iloc[0]
    bottom_row = result_df.iloc[-1]

    summary = (
        f"Para **{metric_label}** agrupado por **{dim_label}**, o destaque"
        f" principal é **{top_row[dim_label]}** ({top_row[metric_label]:,.2f}),"
        f" enquanto o menor índice aparece em **{bottom_row[dim_label]}**"
        f" ({bottom_row[metric_label]:,.2f})."
    )

    # Integração opcional com OpenAI se a chave estiver configurada nos Secrets
    api_key = os.getenv("OPENAI_API_KEY")
    llm_used = False
    if api_key:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            prompt = (
                f"Dados agregados (sem PII):\n{result_df.to_markdown(index=False)}\n\n"
                f"Pergunta do executivo: {question}\n"
                "Responda em 2 frases objetivas em português com foco em decisão de negócios."
            )
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            summary = resp.choices[0].message.content
            llm_used = True
        except Exception:
            pass

    return {
        "summary": summary,
        "table": result_df,
        "x_col": dim_label,
        "y_col": metric_label,
        "llm_used": llm_used,
        "governance_payload": {
            "lgpd_enforced": lgpd_active,
            "excluded_pii_columns": pii_cols if lgpd_active else [],
            "columns_exposed_to_engine": safe_cols,
        },
    }