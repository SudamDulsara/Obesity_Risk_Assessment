MISSING_VALUE_POLICY = {
    "drop_rows": ["id", "NObeyesdad"],
    "fill_zero": ["family_history_with_overweight", "FAVC", "SMOKE", "SCC"],
    "fill_unknown": ["Gender", "CAEC", "CALC", "MTRANS"],
    "fill_null": [],
    "fill_mean": ["CH2O", "FAF", "TUE"],
    "fill_median": ["Age", "Height", "Weight", "FCVC", "NCP"]
}

# Columns you want to rename -> new standardized names
COLUMN_RENAME_MAP = {
    "family_history_with_overweight":"family_hist",
    "FAVC": "highcalorie",
    "FCVC": "vegtables",
    "NCP": "main_meals",
    "CAEC": "snacks",
    "SMOKE": "smokes",
    "CH2O": "water_intake",
    "SCC": "monitors_calories",
    "FAF": "physical_activity",
    "TUE": "screen_time",
    "CALC": "alcohol",
    "MTRANS": "transport",
    "Height": "height_m",
    "Weight": "weight_kg",
    "0be1dad": "obesity_class"
}

WORD_RENAME_MAP = {
    "0rmal_Weight": "Normal_Weight",
}

# --- Validation settings ---

VALIDATION = {
    # Which column is the row identifier and the label?
    "ID_COLUMN": "id",
    "LABEL_COLUMN": "0be1dad",  

    # Columns that must exist (before renaming)
    "REQUIRED_COLUMNS": [
        "Gender","Age","Height","Weight","family_history_with_overweight","FAVC","FCVC","NCP",
        "CAEC","SMOKE","CH2O","SCC","FAF","TUE","CALC","MTRANS","0be1dad"
    ],

    # Binary 0/1 columns
    "BINARY_COLUMNS": ["family_history_with_overweight", "FAVC", "SMOKE", "SCC"],

    # Allowed values for categoricals
    "ENUM_MAP": {
        "Gender": ["Male", "Female"],
        "CAEC": ["No", "Sometimes", "Frequently", "Always"],
        "CALC": ["No", "Sometimes", "Frequently"],
        "MTRANS": ["Public_Transportation", "Automobile", "Walking", "Bike", "Motorbike"]
    },

    # Numeric ranges (min, max) — keep these generous
    "RANGE_MAP": {
        "Age":   (10, 90),
        "Height":(1.30, 2.20),
        "Weight":(30, 250),
        "FCVC":  (1, 3),
        "NCP":   (1, 4),
        "CH2O":  (0, 5),
        "FAF":   (0, 3),
        "TUE":   (0, 2),
    },

    # Heuristics for reduction hints
    "NEAR_CONSTANT_FREQ": 0.98,     # flag columns where top value ≥ 98%
    "RARE_CATEGORY_MIN_COUNT": 10,  # flag categories with < 10 rows
}

    # Map column name -> number of decimal places to keep (0 = whole numbers)
NUMERIC_PRECISION = {
    # examples (tweak or leave empty to disable per column):
    "Age": 0,
    "water_intake": 4,            # make whole numbers
    "vegtables": 4,
    "main_meals": 4,
    "physical_activity": 4,
    "screen_time" :4,
    "BMI": 2,
}

REDUCTION = {
    # 0) Optional row downsampling for dev speed (set None to keep all)
    #"ROW_SAMPLE_MAX": 10000,            # e.g., 50000 to cap rows
    "ROW_SAMPLE_STRATIFY": True,       # keep class balance if label present

    # 1) Drop explicit columns (IDs, free text you don't use, etc.)
    #"DROP_COLUMNS": ["id"],

    # 2) Create BMI and optionally drop sources
    "CREATE_BMI": True,
    "BMI_HEIGHT_COL": "height",
    "BMI_WEIGHT_COL": "weight",
    "BMI_NEW_COL": "BMI",
    "BMI_DROP_SOURCE": True,           # drop Height/Weight after creating BMI

    # 3) Remove near-constant columns (top value frequency ≥ threshold)
    "NEAR_CONSTANT_FREQ": 0.99,

    # 4) Remove highly correlated numeric columns (keep one of each pair)
    "HIGH_CORR_THRESHOLD": 0.92,
    #"PROTECT_COLUMNS": ["obesity_class"],  # never drop these

    # 5) Pool rare categories to avoid one-hot explosion
    "RARE_CATEGORY_MIN_COUNT": 20,
    "RARE_CATEGORY_POOL_VALUE": "Other",
    "RARE_CATEGORY_COLUMNS": [], 
}

