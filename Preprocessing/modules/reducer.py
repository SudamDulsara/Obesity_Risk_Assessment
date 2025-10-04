# modules/reducer.py
from __future__ import annotations
import pandas as pd
import numpy as np

def _first_present(df: pd.DataFrame, names):
    for n in names:
        if n in df.columns:
            return n
    return None

def stratified_downsample(df: pd.DataFrame, max_rows: int, label_col: str | None, stratify: bool):
    if max_rows is None or len(df) <= max_rows:
        return df, None
    if stratify and label_col and label_col in df.columns:
        # stratified sample by label
        frac = max_rows / len(df)
        df_small = (
            df.groupby(label_col, group_keys=False)
              .apply(lambda g: g.sample(max(int(round(len(g) * frac)), 1), random_state=42))
        )
        info = {"from": int(len(df)), "to": int(len(df_small)), "mode": "stratified", "by": label_col}
        return df_small, info
    else:
        df_small = df.sample(max_rows, random_state=42)
        info = {"from": int(len(df)), "to": int(len(df_small)), "mode": "random"}
        return df_small, info

def drop_columns(df: pd.DataFrame, cols):
    cols = [c for c in (cols or []) if c in df.columns]
    return df.drop(columns=cols), cols

def create_bmi(df: pd.DataFrame, height_col: str, weight_col: str, new_col: str, drop_source: bool):
    # Be forgiving: try common fallbacks if the specified names are missing
    h = _first_present(df, [height_col, "Height", "height", "height_m"])
    w = _first_present(df, [weight_col, "Weight", "weight", "weight_kg"])
    if h is None or w is None:
        return df, {"created": False, "used": None, "dropped": []}
    df = df.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        df[new_col] = df[w] / (df[h] ** 2)
    dropped = []
    if drop_source:
        for c in {h, w}:
            if c in df.columns:
                df = df.drop(columns=c)
                dropped.append(c)
    return df, {"created": True, "used": {"height": h, "weight": w, "new": new_col}, "dropped": dropped}

def drop_near_constant(df: pd.DataFrame, threshold: float, protect: set[str]):
    dropped = {}
    n = len(df)
    for col in df.columns:
        if col in protect: 
            continue
        vc = df[col].value_counts(dropna=False)
        if vc.empty: 
            continue
        ratio = vc.iloc[0] / max(1, n)
        if ratio >= threshold:
            dropped[col] = float(ratio)
    return df.drop(columns=list(dropped.keys())), dropped

def pool_rare_categories(df: pd.DataFrame, columns, min_count: int, pool_value: str):
    applied = {}
    df2 = df.copy()
    for col in (columns or []):
        if col not in df2.columns:
            continue
        if not pd.api.types.is_object_dtype(df2[col]) and not pd.api.types.is_string_dtype(df2[col]) and df2[col].dtype.name != "category":
            continue
        vc = df2[col].value_counts(dropna=False)
        rare = vc[vc < min_count]
        if len(rare) == 0:
            continue
        rare_values = set(rare.index.tolist())
        df2[col] = df2[col].where(~df2[col].isin(rare_values), other=pool_value)
        applied[col] = {str(k): int(v) for k, v in rare.items()}
    return df2, applied

def drop_high_correlation(df: pd.DataFrame, threshold: float, protect: set[str]):
    num = df.select_dtypes(include=[np.number]).copy()
    # Don't drop protected or label columns
    keep_mask = [c for c in num.columns if c not in protect]
    num = num[keep_mask]
    if num.shape[1] <= 1:
        return df, {"dropped": [], "pairs": []}

    corr = num.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = set()
    pairs = []
    for col in upper.columns:
        high = upper.index[upper[col] > threshold].tolist()
        for h in high:
            # mark the current column for drop (simple, deterministic)
            if col not in protect and h not in protect:
                to_drop.add(col)
                pairs.append((h, col, float(upper.loc[h, col])))
    df2 = df.drop(columns=list(to_drop), errors="ignore")
    return df2, {"dropped": sorted(list(to_drop)), "pairs": pairs}

def reduce_dataset(df: pd.DataFrame, cfg: dict, label_col: str | None):
    report = {}

    # optional row sampling
    df, sample_info = stratified_downsample(
        df,
        cfg.get("ROW_SAMPLE_MAX"),
        label_col=label_col,
        stratify=bool(cfg.get("ROW_SAMPLE_STRATIFY", True))
    )
    if sample_info:
        report["downsampled"] = sample_info

    # explicit drops
    df, dropped_explicit = drop_columns(df, cfg.get("DROP_COLUMNS", []))
    report["dropped_explicit"] = dropped_explicit

    # BMI
    if cfg.get("CREATE_BMI", False):
        df, bmi_info = create_bmi(
            df,
            cfg.get("BMI_HEIGHT_COL", "Height"),
            cfg.get("BMI_WEIGHT_COL", "Weight"),
            cfg.get("BMI_NEW_COL", "BMI"),
            bool(cfg.get("BMI_DROP_SOURCE", True)),
        )
        report["bmi"] = bmi_info

    # near-constant
    df, nc = drop_near_constant(
        df,
        float(cfg.get("NEAR_CONSTANT_FREQ", 0.99)),
        protect=set(cfg.get("PROTECT_COLUMNS", [])) | ({label_col} if label_col else set()),
    )
    report["near_constant_dropped"] = nc  # {col: ratio}

    # rare categories
    df, pooled = pool_rare_categories(
        df,
        cfg.get("RARE_CATEGORY_COLUMNS", []),
        int(cfg.get("RARE_CATEGORY_MIN_COUNT", 20)),
        cfg.get("RARE_CATEGORY_POOL_VALUE", "Other"),
    )
    report["rare_pooled"] = pooled  # {col: {cat: count}}

    # high correlation
    df, hc = drop_high_correlation(
        df,
        float(cfg.get("HIGH_CORR_THRESHOLD", 0.92)),
        protect=set(cfg.get("PROTECT_COLUMNS", [])) | ({label_col} if label_col else set()),
    )
    report["high_corr"] = hc  # {"dropped":[...], "pairs":[(kept, dropped, corr), ...]}

    return df, report
