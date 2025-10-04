import pandas as pd
from config.settings import WORD_RENAME_MAP

def standardize_columns(df, rename_map: dict):
    """
    Rename columns using a provided mapping. Returns:
      df2: DataFrame with renamed columns
      applied: {old_name: new_name} actually applied (present in df and different)
    """
    applied = {
        old: new for old, new in rename_map.items()
        if old in df.columns and old != new
    }
    df2 = df.rename(columns=applied)
    return df2, applied



def apply_word_rename(df: pd.DataFrame, rename_map: dict):

    report = {}
    if not rename_map:
        return df, report

    for col in df.columns:
        # Only apply to string columns (skip others)
        if pd.api.types.is_string_dtype(df[col]):
            before = df[col].copy()
            # Replace all mapped words
            df[col] = df[col].replace(rename_map, regex=True)

            # Count changes made in this column
            changes = (before != df[col]).sum()
            if changes > 0:
                report[col] = changes

    return df, report
