import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

CSV_PATH = "./Preprocessing/output/cleaned_obesity_level.csv"
TARGET = "obesity_class"
DROP_COLS = ["id"]
RANDOM_STATE = 42
N_FOLDS = 5

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    numeric_preprocess = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_preprocess = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", ohe),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_preprocess, numeric_features),
            ("cat", categorical_preprocess, categorical_features),
        ]
    )
    return preprocess

def main():
    df = pd.read_csv(CSV_PATH)

    for c in DROP_COLS:
        if c in df.columns:
            df = df.drop(columns=c)

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in CSV. Columns: {list(df.columns)}")

    y = df[TARGET]
    X = df.drop(columns=[TARGET])

    preprocess = build_preprocessor(X)

    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    knn = KNeighborsClassifier(n_neighbors=7, weights="distance")

    estimators = [
        ("lr", lr),
        ("rf", rf),
        ("knn", knn),
    ]

    final_est = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)

    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=final_est,
        stack_method="auto",
        passthrough=False,
        cv=StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
    )

    pipe = Pipeline(steps=[("prep", preprocess), ("stack", stack)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)

    print("=== Stacking Ensemble (LR + RF + KNN) ===")
    print(f"Train CV Accuracy (mean±std over {N_FOLDS} folds): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    sys.exit(main())
