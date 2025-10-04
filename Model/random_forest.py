import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
import numpy as np
import json, sys, joblib, sklearn
from pathlib import Path 

CSV_PATH = "./Preprocessing/output/cleaned_obesity_level.csv"
TARGET = "obesity_class"
DROP_COLS = ["id"]
RANDOM_STATE = 42

OUTPUT_DIR = Path.cwd() / "Trained"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = OUTPUT_DIR / "RF_model.pkl"
META_PATH = OUTPUT_DIR / "RF_model_meta.json"

df = pd.read_csv(CSV_PATH)

for c in DROP_COLS:
    if c in df.columns:
        df = df.drop(columns=c)

y = df[TARGET]
X = df.drop(columns=[TARGET])

numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = [c for c in X.columns if c not in numeric_features]

try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

numeric_preprocess = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median"))]
)

categorical_preprocess = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", ohe)]
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_preprocess, numeric_features),
        ("cat", categorical_preprocess, categorical_features),
    ]
)

clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

pipe = Pipeline(steps=[("prep", preprocess), ("model", clf)])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


def _final_estimator(p):
    return p.steps[-1][1] if isinstance(p, Pipeline) else p

def _extract_cat_categories(fitted_pipe, categorical_cols):
    cats = {}
    try:
        ct = None
        if isinstance(fitted_pipe, Pipeline):
            for _, step in fitted_pipe.steps:
                if isinstance(step, ColumnTransformer):
                    ct = step
                    break
        if ct is None:
            return cats

        for name, transformer, cols in ct.transformers_:
            est = transformer
            if isinstance(est, Pipeline):
                for _, sub in est.steps:
                    if isinstance(sub, OneHotEncoder):
                        est = sub
                        break
            if isinstance(est, OneHotEncoder) and hasattr(est, "categories_"):
                for i, col in enumerate(cols):
                    if col in categorical_cols:
                        cats[col] = [str(v) for v in est.categories_[i]]
    except Exception:
        pass
    return cats

joblib.dump(pipe, MODEL_PATH)

final_est = _final_estimator(pipe)
classes = getattr(final_est, "classes_", None)
cat_map = _extract_cat_categories(pipe, list(categorical_features))

meta = {
    "feature_order": list(X.columns),
    "numeric_features": list(numeric_features),
    "categorical_features": list(categorical_features),
    "categorical_values": cat_map,
    "classes": list(map(str, classes)) if classes is not None else None,
    "versions": {"python": sys.version, "sklearn": sklearn.__version__}
}

with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print(f"Saved {MODEL_PATH} and {META_PATH}")