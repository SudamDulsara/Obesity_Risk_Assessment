import os
import pandas as pd
from config.settings import MISSING_VALUE_POLICY, COLUMN_RENAME_MAP, VALIDATION, REDUCTION, NUMERIC_PRECISION, WORD_RENAME_MAP
from modules.cleaner import remove_duplicates, handle_missing_values
from modules.transformer import standardize_columns, apply_word_rename
from modules.validator import validate_dataframe, apply_numeric_precision
from modules.reducer import reduce_dataset

# Paths
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

def load_data(filename: str):
    filepath = os.path.join(DATA_DIR, filename)
    if filename.endswith(".csv"):
        return pd.read_csv(filepath)
    elif filename.endswith(".xlsx"):
        return pd.read_excel(filepath)
    else:
        raise ValueError("Unsupported file format. Use CSV or XLSX.")

def _sum_report_counts(report: dict) -> int:
    total = 0
    for k, v in report.items():
        if k == "drop_rows":
            continue
        if isinstance(v, dict):
            total += sum(int(n) for n in v.values())
        else:
            total += int(v or 0)
    return total

def main():
    # Change filename here
    filename = "obesity_level.csv"
    df = load_data(filename)
    print("Data loaded successfully!")

    # Validation
    v = validate_dataframe(df, VALIDATION)

    if v["schema"]["missing_required"]:
        print(" Missing required columns:", ", ".join(v["schema"]["missing_required"]))
    if v["missing"]:
        print(" Missing values:", ", ".join(f"{c}={n}" for c, n in v["missing"].items()))
    if v["duplicate_rows"]:
        print(f" Duplicate rows (pre-clean): {v['duplicate_rows']}")
    if v["id_checks"]["present"]:
        if v["id_checks"]["missing"] or v["id_checks"]["duplicates"]:
            print(f" ID issues → missing: {v['id_checks']['missing']}, duplicates: {v['id_checks']['duplicates']}")
    if v["enum_violations"]:
        print(" Enum violations:", ", ".join(f"{c}({d['count']})" for c, d in v["enum_violations"].items()))
    if v["range_violations"]:
        print(" Range violations:", ", ".join(f"{c}({d['count']})" for c, d in v["range_violations"].items()))
    if v["near_constant"]:
        print(" Near-constant:", ", ".join(f"{c}({d['ratio']:.2f})" for c, d in v["near_constant"].items()))
    if v["rare_categories"]:
        print(" Rare categories:", ", ".join(v["rare_categories"].keys()))

    # 1) remove duplicates
    df, dup_removed = remove_duplicates(df)

    # 2) handle missing values per settings
    df, mv_report = handle_missing_values(df, MISSING_VALUE_POLICY)

    # 3) standardize column names per settings
    df, rename_report = standardize_columns(df, COLUMN_RENAME_MAP)

    # Apply word renaming
    df, word_rename_report = apply_word_rename(df, WORD_RENAME_MAP)
    if word_rename_report:
        changed_cols = ", ".join(f"{c}={n}" for c, n in word_rename_report.items())
        print(f" Word normalization applied: {changed_cols}")
    else:
        print(" Word normalization: none (no matches).")

    # 4) reduce dataset (rows/columns) per settings
    # Determine current label name (after renaming)
    label_original = VALIDATION.get("LABEL_COLUMN")
    label_current = COLUMN_RENAME_MAP.get(label_original, label_original)
    df, red = reduce_dataset(df, REDUCTION, label_col=label_current)

    # Round off values
    df, precision_report = apply_numeric_precision(df, NUMERIC_PRECISION)
    if precision_report:
        changed_cols = ", ".join(f"{c}={n}" for c, n in precision_report.items())
        print(f" Numeric precision applied: {changed_cols}")
    else:
        print(" Numeric precision: none (no rules or no matching columns).")
        
    # 5) save once
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(
        OUTPUT_DIR, "cleaned_" + os.path.splitext(filename)[0] + ".csv"
    )
    df.to_csv(output_file, index=False)

    # 6) minimal reporting
    print(f" Removed duplicate rows: {dup_removed}")
    dropped = mv_report.get("drop_rows", 0)
    dropped_rows = dropped.get("_rows_dropped", 0) if isinstance(dropped, dict) else int(dropped or 0)
    print(f" Rows dropped due to missing (drop_rows): {dropped_rows}")

    total_filled = _sum_report_counts(mv_report)
    print(f" Missing values filled (total): {total_filled}")
    for step, detail in mv_report.items():
        if step == "drop_rows":
            continue
        if isinstance(detail, dict):
            step_total = sum(int(n) for n in detail.values())
            if step_total:
                cols = ", ".join(f"{c}={n}" for c, n in detail.items() if n)
                print(f"  - {step}: {step_total} ({cols})")

    if rename_report:
        print(f" Standardized columns ({len(rename_report)}):")
        for old, new in rename_report.items():
            print(f"  - {old} → {new}")
    else:
        print(" Standardized columns: none (mapping did not match any current columns).")

    # Reduction summary
    if red.get("downsampled"):
        ds = red["downsampled"]
        mode = ds["mode"]
        by = f" by {ds['by']}" if "by" in ds else ""
        print(f" Rows: {ds['from']} → {ds['to']} ({mode}{by})")
    if red.get("dropped_explicit"):
        print(f" Dropped explicit cols: {', '.join(red['dropped_explicit'])}")
    if red.get("bmi", {}).get("created"):
        used = red["bmi"]["used"]
        print(f" BMI created from {used['height']} & {used['weight']}.", end="")
        if red["bmi"]["dropped"]:
            print(f" Dropped: {', '.join(red['bmi']['dropped'])}")
        else:
            print()
    if red.get("near_constant_dropped"):
        cols = ", ".join(f"{c}({r:.2f})" for c, r in red["near_constant_dropped"].items())
        if cols:
            print(f"⚖️ Near-constant dropped: {cols}")
    if red.get("rare_pooled"):
        for col, cats in red["rare_pooled"].items():
            if cats:
                cats_str = ", ".join(f"{k}={v}" for k, v in cats.items())
                print(f" Pooled rare in {col}: {cats_str}")
    if red.get("high_corr", {}).get("dropped"):
        print(f" High-corr dropped (> {REDUCTION.get('HIGH_CORR_THRESHOLD', 0.92)}): "
              f"{', '.join(red['high_corr']['dropped'])}")


    print(f" Cleaned file saved at: {output_file}")

if __name__ == "__main__":
    main()
