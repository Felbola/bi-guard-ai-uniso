import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge


def forecast_monthly_revenue(
    df: pd.DataFrame, periods_ahead: int = 3
) -> pd.DataFrame:
    monthly = (
        df.assign(month=df["date"].dt.to_period("M").dt.to_timestamp())
        .groupby("month", as_index=False)["revenue"]
        .sum()
        .sort_values("month")
        .reset_index(drop=True)
    )

    if len(monthly) < 3:
        monthly["type"] = "Histórico"
        return monthly

    monthly["t"] = np.arange(len(monthly))
    # Feature cíclica para capturar sazonalidade trimestral
    monthly["sin_q"] = np.sin(2 * np.pi * monthly["month"].dt.month / 3.0)
    monthly["cos_q"] = np.cos(2 * np.pi * monthly["month"].dt.month / 3.0)

    features = ["t", "sin_q", "cos_q"]
    model = Ridge(alpha=1.0)
    model.fit(monthly[features], monthly["revenue"])

    last_date = monthly["month"].max()
    future_dates = [
        last_date + pd.DateOffset(months=i) for i in range(1, periods_ahead + 1)
    ]
    future_t = np.arange(len(monthly), len(monthly) + periods_ahead)

    future_df = pd.DataFrame({
        "month": future_dates,
        "t": future_t,
        "sin_q": [np.sin(2 * np.pi * d.month / 3.0) for d in future_dates],
        "cos_q": [np.cos(2 * np.pi * d.month / 3.0) for d in future_dates],
    })
    future_df["revenue"] = np.maximum(
        0, model.predict(future_df[features])
    ).round(2)
    future_df["type"] = "Projeção (Ridge ML)"

    monthly["type"] = "Histórico"
    return pd.concat(
        [
            monthly[["month", "revenue", "type"]],
            future_df[["month", "revenue", "type"]],
        ],
        ignore_index=True,
    )


def detect_transaction_anomalies(
    df: pd.DataFrame, contamination: float = 0.03
) -> pd.DataFrame:
    if len(df) < 20:
        return pd.DataFrame()

    features = df[["revenue", "churn_probability"]].copy()
    iso = IsolationForest(contamination=contamination, random_state=42)
    preds = iso.fit_predict(features)

    anomalies = df[preds == -1].copy()
    return anomalies.sort_values("revenue", ascending=False)
