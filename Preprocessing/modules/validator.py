# modules/validator.py
from collections import defaultdict
import pandas as pd
import numpy as np

def _value_counts_safe(s: pd.Series):
    try:
        return s.value_counts(dropna=False)
    except Exception:
        return pd.Series(dtype="int64")

def validate_dataframe(df: pd.DataFrame, cfg: dict):
    """
    Returns a structured report (dict) with:
      - rows/cols
      - schema issues (missing required columns)
      - missing counts per column
      - duplicate rows count
      - id column: missing/duplicate counts
      - enum violations per categorical column
      - numeric range violations per numeric column
      - binary violations for specified columns
      - near-constant columns
      - rare categories (per categorical)
    """
    report = {
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "schema": {"missing_required": []},
        "missing": {},
        "duplicate_rows": int(df.duplicated().sum()),
        "id_checks": {"present": False, "missing": 0, "duplicates": 0},
        "enum_violations": {},      # col -> {"count": int, "invalid": [..]}
        "range_violations": {},     # col -> {"count": int}
        "binary_violations": {},    # col -> {"count": int, "invalid": [..]}
        "near_constant": {},        # col -> {"top_value": x, "ratio": float}
        "rare_categories": {},      # col -> {category: count}
    }

    # --- schema: required columns present? ---
    required = cfg.get("REQUIRED_COLUMNS", [])
    missing_required = [c for c in required if c not in df.columns]
    report["schema"]["missing_required"] = missing_required

    # --- missing per column (NaN only; whitespace handled in cleaning) ---
    report["missing"] = {c: int(n) for c, n in df.isna().sum().items() if int(n) > 0}

    # --- id checks ---
    id_col = cfg.get("ID_COLUMN")
    if id_col in df.columns:
        report["id_checks"]["present"] = True
        report["id_checks"]["missing"] = int(df[id_col].isna().sum())
        report["id_checks"]["duplicates"] = int(df[id_col].duplicated().sum())

    # --- enum checks for categoricals ---
    enum_map = cfg.get("ENUM_MAP", {})
    for col, allowed in enum_map.items():
        if col not in df.columns:
            continue
        s = df[col].astype("string")
        invalid_mask = ~s.isna() & ~s.isin(pd.Series(allowed, dtype="string"))
        cnt = int(invalid_mask.sum())
        if cnt > 0:
            invalid_vals = list(s[invalid_mask].value_counts().head(10).index)
            report["enum_violations"][col] = {"count": cnt, "invalid": invalid_vals}

    # --- numeric range checks ---
    range_map = cfg.get("RANGE_MAP", {})
    for col, (lo, hi) in range_map.items():
        if col not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            mask = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
            cnt = int(mask.sum())
            if cnt > 0:
                report["range_violations"][col] = {"count": cnt}
        else:
            # not numeric but expected numeric
            non_num = int(df[col].notna().sum())
            if non_num > 0:
                report["range_violations"][col] = {"count": non_num, "note": "non-numeric values"}

    # --- binary columns strictly in {0,1} ---
    for col in cfg.get("BINARY_COLUMNS", []):
        if col not in df.columns:
            continue
        invalid_mask = df[col].notna() & ~df[col].isin([0, 1])
        cnt = int(invalid_mask.sum())
        if cnt > 0:
            invalid_vals = list(_value_counts_safe(df.loc[invalid_mask, col]).head(10).index)
            report["binary_violations"][col] = {"count": cnt, "invalid": invalid_vals}

    # --- near-constant columns (top value frequency) ---
    thresh = float(cfg.get("NEAR_CONSTANT_FREQ", 0.98))
    for col in df.columns:
        vc = _value_counts_safe(df[col])
        if vc.empty:
            continue
        top = int(vc.iloc[0])
        ratio = top / max(1, len(df))
        if ratio >= thresh:
            report["near_constant"][col] = {"top_value": vc.index[0], "ratio": round(ratio, 4)}

    # --- rare categories (categoricals) ---
    min_count = int(cfg.get("RARE_CATEGORY_MIN_COUNT", 10))
    # consider enum columns and any object/string columns as categorical
    cat_cols = set(enum_map.keys()) | set(df.select_dtypes(include=["object", "string", "category"]).columns)
    for col in cat_cols:
        if col not in df.columns:
            continue
        vc = _value_counts_safe(df[col])
        rare = vc[vc < min_count]
        if len(rare) > 0:
            report["rare_categories"][col] = {str(k): int(v) for k, v in rare.items()}

    return report

def apply_numeric_precision(df: pd.DataFrame, precision_map: dict):
    """
    Rounds numeric columns to the specified number of decimal places.
    precision_map: { "col_name": int_decimals, ... }
      - 0 decimals => casts to pandas nullable Int64 for clean whole numbers.
    Returns: (df, report) where report is {col: changed_values_count}
    """
    report = {}
    if not precision_map:
        return df, report

    for col, dec in precision_map.items():
        if col not in df.columns:
            continue  # skip silently if column isn't present

        # ensure numeric (coerce non-numeric to NaN)
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        before = df[col].copy()
        df[col] = df[col].round(int(dec))

        if int(dec) == 0:
            # cast to Int64 so you don't get ".0" in outputs
            try:
                df[col] = df[col].astype("Int64")
            except Exception:
                # if cast fails (e.g., all NaN), leave as-is
                pass

        # count changed values (ignore NaNs)
        mask_before = before.notna()
        mask_after = df[col].notna()
        compared = mask_before & mask_after
        changed = int((before[compared] != df[col][compared]).sum())
        report[col] = changed

    return df, report
