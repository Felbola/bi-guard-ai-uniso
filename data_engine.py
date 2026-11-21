import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=False)
def generate_enterprise_dataset(n_records: int = 1200) -> pd.DataFrame:
    rng = np.random.default_rng(seed=2026)

    dates = pd.date_range(start="2025-01-01", end="2026-10-31", freq="D")
    sampled_dates = rng.choice(dates, size=n_records)

    regions = ["Sudeste", "Sul", "Nordeste", "Centro-Oeste", "Norte"]
    region_weights = [0.44, 0.22, 0.18, 0.10, 0.06]

    categories = [
        "Cloud & Infra",
        "Licenciamento SaaS",
        "Consultoria Dados",
        "Segurança Cibernética",
    ]
    base_prices = {
        "Cloud & Infra": 3200.0,
        "Licenciamento SaaS": 1850.0,
        "Consultoria Dados": 4500.0,
        "Segurança Cibernética": 3900.0,
    }

    first_names = [
        "Lucas",
        "Mariana",
        "Rafael",
        "Camila",
        "Bruno",
        "Fernanda",
        "Thiago",
        "Juliana",
        "Gabriel",
        "Patricia",
    ]
    last_names = [
        "Almeida",
        "Costa",
        "Ferreira",
        "Oliveira",
        "Santos",
        "Rodrigues",
        "Pereira",
        "Lima",
        "Gomes",
        "Martins",
    ]

    chosen_cats = rng.choice(categories, size=n_records)
    chosen_dates = pd.to_datetime(sampled_dates)

    # Aplica tendência de crescimento anual + sazonalidade de fim de trimestre
    months_elapsed = (chosen_dates.year - 2025) * 12 + chosen_dates.month
    trend_factor = 1 + (months_elapsed * 0.015)
    quarter_boost = np.where(chosen_dates.month.isin([3, 6, 9, 12]), 1.18, 1.0)

    revenues = [
        round(
            max(
                250.0,
                rng.normal(base_prices[cat], base_prices[cat] * 0.28)
                * tf
                * qb,
            ),
            2,
        )
        for cat, tf, qb in zip(chosen_cats, trend_factor, quarter_boost)
    ]

    full_names = [
        f"{rng.choice(first_names)} {rng.choice(last_names)}"
        for _ in range(n_records)
    ]
    cpfs = [
        (
            f"{rng.integers(100,999)}.{rng.integers(100,999)}."
            f"{rng.integers(100,999)}-{rng.integers(10,99)}"
        )
        for _ in range(n_records)
    ]
    emails = [
        f"{name.lower().replace(' ', '.')}{rng.integers(1,99)}@corp.com.br"
        for name in full_names
    ]

    df = pd.DataFrame({
        "transaction_id": [f"ORD-{20260000 + i}" for i in range(n_records)],
        "date": chosen_dates,
        "customer_name": full_names,
        "cpf": cpfs,
        "email": emails,
        "region": rng.choice(regions, size=n_records, p=region_weights),
        "category": chosen_cats,
        "revenue": revenues,
        "churn_probability": np.round(rng.beta(2, 5, size=n_records), 3),
    })

    return df.sort_values("date").reset_index(drop=True)


def load_uploaded_csv(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df
