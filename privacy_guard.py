import hashlib
import re
import numpy as np
import pandas as pd

PII_PATTERNS = re.compile(
    r"(name|nome|cpf|rg|email|e-mail|telefone|phone|endereco|address)", re.I
)


def detect_pii_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if PII_PATTERNS.search(col)]


def hash_identifier(value: str, salt: str = "UNISO_BI_2026") -> str:
    digest = hashlib.sha256(f"{salt}:{value}".encode("utf-8")).hexdigest()
    return f"PSEUDO-{digest[:10].upper()}"


def mask_value(col_name: str, val: str) -> str:
    val_str = str(val)
    col_lower = col_name.lower()
    if "cpf" in col_lower and len(val_str) >= 11:
        return f"***.***.{val_str[-6:]}"
    if "email" in col_lower and "@" in val_str:
        user, domain = val_str.split("@", 1)
        return f"{user[:2]}***@{domain}"
    if len(val_str) > 2:
        return f"{val_str[0]}{'*' * (len(val_str) - 2)}{val_str[-1]}"
    return "***"


def apply_privacy_pipeline(
    df: pd.DataFrame,
    enabled: bool = True,
    epsilon: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Aplica pseudonimização, mascaramento de PII e Privacidade Diferencial (Mecanismo de Laplace).
    Quanto menor o epsilon (ε), maior a privacidade e maior o ruído estatístico.
    """
    processed = df.copy()
    pii_cols = detect_pii_columns(processed)

    audit_log = {
        "lgpd_active": enabled,
        "detected_pii_columns": pii_cols,
        "differential_privacy_epsilon": epsilon,
        "records_processed": len(processed),
    }

    if not enabled:
        return processed, audit_log

    # Gera chave pseudonimizada a partir do primeiro identificador PII disponível
    if pii_cols:
        primary_id = pii_cols[0]
        processed.insert(
            1,
            "subject_token",
            processed[primary_id].astype(str).apply(hash_identifier),
        )

    for col in pii_cols:
        processed[col] = processed[col].apply(lambda x: mask_value(col, x))

    # Mecanismo de Laplace com semente fixa para estabilidade de sessão
    if epsilon is not None and epsilon > 0 and "revenue" in processed.columns:
        rng = np.random.default_rng(seed=int(epsilon * 1000))
        sensitivity = (
            processed["revenue"].quantile(0.95)
            - processed["revenue"].quantile(0.05)
        ) * 0.05
        scale = sensitivity / epsilon
        noise = rng.laplace(loc=0.0, scale=scale, size=len(processed))
        processed["revenue"] = np.maximum(
            0.0, np.round(processed["revenue"] + noise, 2)
        )

    return processed, audit_log
