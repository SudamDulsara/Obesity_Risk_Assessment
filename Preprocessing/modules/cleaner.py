import os

def remove_duplicates(df, output_path: str):
    # Count duplicates
    before = len(df)
    df_cleaned = df.drop_duplicates()
    after = len(df_cleaned)
    removed = before - after

    # Save cleaned data
    df_cleaned.to_csv(output_path, index=False)

    return removed, output_path

import pandas as pd

def remove_duplicates(df):
    before = len(df)
    df2 = df.drop_duplicates()
    removed = before - len(df2)
    return df2, removed

def _standardize_missing(df):
    # Convert empty strings / whitespace to proper NA
    return df.replace(r"^\s*$", pd.NA, regex=True)

def drop_rows_with_missing(df, columns):
    before = len(df)
    df2 = df.dropna(subset=[c for c in columns if c in df.columns])
    dropped = before - len(df2)
    return df2, dropped

def fill_with_zero(df, columns):
    counts = {}
    for col in columns:
        if col in df.columns:
            n = int(df[col].isna().sum())
            df[col] = df[col].fillna(0)
            counts[col] = n
    return df, counts

def fill_with_unknown(df, columns):
    counts = {}
    for col in columns:
        if col in df.columns:
            n = int(df[col].isna().sum())
            df[col] = df[col].fillna("unknown")
            counts[col] = n
    return df, counts

def fill_with_null(df, columns):
    # Useful to standardize to a single null sentinel (pd.NA)
    counts = {}
    for col in columns:
        if col in df.columns:
            n = int(df[col].isna().sum())
            df[col] = df[col].where(~df[col].isna(), other=pd.NA)
            counts[col] = n
    return df, counts

def fill_with_mean(df, columns):
    counts = {}
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            n = int(df[col].isna().sum())
            if n > 0:
                df[col] = df[col].fillna(df[col].mean())
            counts[col] = n
    return df, counts

def fill_with_median(df, columns):
    counts = {}
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            n = int(df[col].isna().sum())
            if n > 0:
                df[col] = df[col].fillna(df[col].median())
            counts[col] = n
    return df, counts

def handle_missing_values(df, policy):
    df = _standardize_missing(df)
    report = {}

    dropped_cols = policy.get("drop_rows", [])
    df, dropped_rows = drop_rows_with_missing(df, dropped_cols)
    report["drop_rows"] = {"_rows_dropped": int(dropped_rows)} 

    for name, func in [
        ("fill_zero", fill_with_zero),
        ("fill_unknown", fill_with_unknown),
        ("fill_null", fill_with_null),
        ("fill_mean", fill_with_mean),
        ("fill_median", fill_with_median),
    ]:
        cols = policy.get(name, [])
        if cols:
            df, counts = func(df, cols)
            report[name] = int(sum(counts.values()))

    return df, report
