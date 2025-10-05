
import os
import json
import math
from pathlib import Path
import importlib.util
import time

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

# ==========================
# Page / Theme
# ==========================

st.set_page_config(
    page_title="Health Analytics | Obesity Risk Assessment",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PALETTE = {
    "bg": "#e9eef5",
    "surface": "#f5f7fb",
    "panel": "#f6f9fc",
    "card": "#f7f9fc",
    "card_border": "#c7d2e0",
    "grad_start": "#e9eef5",
    "grad_mid": "#e5ebf4",
    "grad_end": "#dde6f2",

    "primary": "#0f766e",
    "primary_dim": "#115e59",
    "accent": "#0369a1",

    "text": "#0b132a",
    "text_muted": "#334155",

    "success": "#15803d",
    "warning": "#b45309",
    "danger":  "#b91c1c",
}

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{visibility: hidden;}}
.stDeployButton {{display: none;}}

html, body, [class*="css"] {{
  font-family: 'Inter', sans-serif;
}}

.stApp {{
  background:
    radial-gradient(1200px 600px at 20% -10%, {PALETTE['grad_end']} 0%, transparent 70%),
    radial-gradient(1200px 600px at 80% 0%, {PALETTE['grad_mid']} 0%, transparent 65%),
    linear-gradient(180deg, {PALETTE['bg']} 0%, {PALETTE['grad_mid']} 100%);
  color: {PALETTE['text']};
}}

.main .block-container {{
  padding: 2rem 1rem;
  max-width: 1200px;
}}

.hero-card {{
  background: linear-gradient(180deg, {PALETTE['surface']} 0%, {PALETTE['card']} 100%);
  border-radius: 20px;
  padding: 2rem 2rem;
  margin-bottom: 1.25rem;
  border: 1px solid {PALETTE['card_border']};
  border-left: 6px solid {PALETTE['primary']};
  box-shadow: 0 10px 28px -20px rgba(2, 6, 23, .35);
}}

.section-header {{
  background: rgba(15,118,110,.06);
  border-radius: 12px;
  padding: 0.9rem 1rem;
  margin: 0.9rem 0 0.75rem 0;
  border: 1px solid {PALETTE['card_border']};
  border-left: 4px solid {PALETTE['primary']};
}}

.section-header h3 {{
  color: {PALETTE['text']};
  font-size: 1.0rem;
  font-weight: 800;
  margin: 0;
  letter-spacing: .01em;
}}

.custom-card {{
  background: {PALETTE['card']};
  border-radius: 14px;
  padding: 1rem;
  border: 1px solid {PALETTE['card_border']};
  box-shadow: 0 6px 16px -14px rgba(2,132,199,.18);
  margin-bottom: 0.9rem;
}}

.result-card {{
  background: {PALETTE['surface']};
  border-radius: 18px;
  padding: 1.25rem;
  border: 1px solid {PALETTE['card_border']};
  box-shadow: 0 10px 24px -18px rgba(15,118,110,.20);
}}

.badge {{
  display:inline-flex; align-items:center; gap:.5rem;
  background: rgba(14,165,164,.10);
  color: {PALETTE['primary_dim']};
  padding:.35rem .7rem; border-radius: 999px;
  border: 1px solid rgba(14,165,164,.35);
  font-weight: 700; font-size:.78rem; letter-spacing:.02em;
}}

.stat-grid {{
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 0.75rem;
}}

.stat-card {{
  background: {PALETTE['card']};
  border-radius: 12px;
  border: 1px solid {PALETTE['card_border']};
  padding: 0.9rem 1rem;
  text-align: center;
}}

.stat-card .k {{
  color: {PALETTE['text_muted']};
  font-size: .78rem; font-weight: 700; text-transform: uppercase; letter-spacing: .03em;
  margin-bottom: .25rem;
}}

.stat-card .v {{
  color: {PALETTE['text']};
  font-size: 1.35rem; font-weight: 800;
}}

.proba-bar {{
  height: 10px; background: #d1d5db; border-radius: 999px; overflow: hidden; border: 1px solid {PALETTE['card_border']};
}}

.proba-fill {{
  height: 100%; background: linear-gradient(90deg, {PALETTE['primary']} 0%, {PALETTE['accent']} 100%);
}}

.stButton button {{
  background: linear-gradient(135deg, {PALETTE['primary']} 0%, {PALETTE['primary_dim']} 100%) !important;
  color: #fff !important;
  border: none !important;
  padding: 0.75rem 1.25rem !important;
  font-size: 1rem !important;
  font-weight: 700 !important;
  border-radius: 10px !important;
  width: 100% !important;
  letter-spacing: .02em !important;
  box-shadow: 0 10px 24px -14px rgba(13,148,136,.40) !important;
  filter: saturate(.95) brightness(.98);
}}

.stButton button * {{
  color: #fff !important;
  fill: #fff !important;
}}

.stButton button:focus {{
  outline: 3px solid #99f6e4 !important; outline-offset: 2px;
}}
.stButton button:hover {{
  filter: saturate(1) brightness(1);
}}

div[data-testid="stFormSubmitButton"] button[data-testid="baseButton-secondary"],
div[data-testid="stFormSubmitButton"] button[data-testid="baseButton-primary"]{{
  background: linear-gradient(135deg, {PALETTE['primary']} 0%, {PALETTE['primary_dim']} 100%) !important;
  color: #fff !important;
  border: none !important;
  box-shadow: 0 10px 24px -14px rgba(13,148,136,.40) !important;
}}

div[data-testid="stFormSubmitButton"] button[data-testid="baseButton-secondary"] span,
div[data-testid="stFormSubmitButton"] button[data-testid="baseButton-primary"] span,
div[data-testid="stFormSubmitButton"] button[data-testid="baseButton-secondary"] svg,
div[data-testid="stFormSubmitButton"] button[data-testid="baseButton-primary"] svg{{
  color: #fff !important;
  fill: #fff !important;
}}

div[data-testid="stFormSubmitButton"] button:disabled{{
  opacity: 1 !important;
}}
div[data-testid="stFormSubmitButton"] button:disabled *{{
  color:#fff !important; fill:#fff !important;
}}

.stApp::before{{
  content:"";
  position:fixed; inset:0; pointer-events:none; opacity:.6;
  background:
    repeating-linear-gradient(0deg,rgba(2,6,23,.02),rgba(2,6,23,.02) 1px,transparent 1px,transparent 24px),
    repeating-linear-gradient(90deg,rgba(2,6,23,.02),rgba(2,6,23,.02) 1px,transparent 1px,transparent 24px);
}}

label{{color:{PALETTE['text_muted']};font-weight:600;}}
[data-testid="stMarkdownContainer"], .small{{color:{PALETTE['text_muted']};}}

.app-footer{{border-top: 1px solid {PALETTE['card_border']}; margin-top:1.2rem; padding:1rem 0; color:{PALETTE['text_muted']}; text-align:center;}}
.app-footer .brand{{font-weight:700;color:{PALETTE['text']};}}
</style>
""",
    unsafe_allow_html=True,
)

# ==========================
# Paths & Config
# ==========================

APP_DIR = Path(__file__).resolve().parent
TRAINED_DIR = APP_DIR / "Trained"
PREPROC_DIR = APP_DIR / "Preprocessing" / "output"
CSV_FALLBACK = PREPROC_DIR / "cleaned_obesity_level.csv"
MODEL_PATH = TRAINED_DIR / "RF_model.pkl"
META_PATHS = [TRAINED_DIR / "RF_model_meta.json", APP_DIR / "RF_model_meta.json"]

TRAINED_DIR.mkdir(parents=True, exist_ok=True)

# ==========================
# Artifact Download Utilities (restored)
# ==========================

def _find_secret_or_env(key: str, default: str | None = None) -> str | None:
    # st.secrets first, then os.environ
    try:
        v = st.secrets.get(key)
        if v:
            return str(v)
    except Exception:
        pass
    return os.environ.get(key, default)

def _extract_drive_id(url_or_id: str | None) -> str | None:
    if not url_or_id:
        return None
    s = str(url_or_id).strip()
    if len(s) >= 20 and "/" not in s and " " not in s and "." not in s:
        return s  # looks like bare file ID
    # common drive formats
    if "drive.google.com" in s and "/d/" in s:
        try:
            return s.split("/d/")[1].split("/")[0]
        except Exception:
            return None
    if "id=" in s:
        try:
            return s.split("id=")[1].split("&")[0]
        except Exception:
            return None
    return None

def _download_with_gdown(url: str | None = None, file_id: str | None = None, output: Path | None = None) -> bool:
    spec = importlib.util.find_spec("gdown")
    if spec is None:
        return False
    import gdown  # type: ignore
    try:
        if file_id:
            final_out = gdown.download(id=file_id, output=str(output) if output else None, quiet=False)
        else:
            final_out = gdown.download(url=url, output=str(output) if output else None, quiet=False)
        return bool(final_out and Path(final_out).exists())
    except Exception:
        return False

def _stream_download(url: str, dest: Path) -> bool:
    # Simple streamed download with one progress bar widget reused
    try:
        with requests.Session() as s:
            r1 = s.get(url, stream=True, timeout=30)
            if r1.status_code != 200:
                return False
            total = int(r1.headers.get("content-length", "0")) or None
            chunk = 1024 * 1024
            prog = st.progress(0.0)
            written = 0
            with open(dest, "wb") as f:
                for b in r1.iter_content(chunk_size=chunk):
                    if not b:
                        continue
                    f.write(b)
                    written += len(b)
                    if total:
                        prog.progress(min(written / total, 1.0))
            if total and written < total:
                return False
        return dest.exists() and dest.stat().st_size > 0
    except Exception:
        return False

def _download_from_google_drive_fallback(file_id: str, dest: Path) -> bool:
    # Minimal confirm-token flow for large Drive files
    try:
        URL = "https://docs.google.com/uc?export=download"
        session = requests.Session()
        response = session.get(URL, params={"id": file_id}, stream=True, timeout=30)
        token = None
        for k, v in response.cookies.items():
            if k.startswith("download_warning"):
                token = v
                break
        params = {"id": file_id}
        if token:
            params["confirm"] = token
        response = session.get(URL, params=params, stream=True, timeout=30)
        total = int(response.headers.get("content-length", "0")) or None
        prog = st.progress(0.0)
        chunk = 1024 * 1024
        written = 0
        with open(dest, "wb") as f:
            for b in response.iter_content(chunk_size=chunk):
                if not b:
                    continue
                f.write(b)
                written += len(b)
                if total:
                    prog.progress(min(written / total, 1.0))
        return dest.exists() and dest.stat().st_size > 0
    except Exception:
        return False

def ensure_artifacts() -> tuple[Path | None, Path | None]:
    """
    Returns (model_path, meta_path) ensuring they exist locally.
    Sources can be provided via:
      - MODEL_URL / MODEL_DRIVE_ID
      - META_URL / META_DRIVE_ID
    """
    model_path = MODEL_PATH
    meta_path = None
    meta_candidates = META_PATHS

    model_ok = model_path.exists() and model_path.stat().st_size > 0
    if not model_ok:
        st.info("Model file not found locally. Attempting download…")
        model_url = _find_secret_or_env("MODEL_URL")
        model_id = _extract_drive_id(_find_secret_or_env("MODEL_DRIVE_ID"))
        if model_url or model_id:
            model_tmp = TRAINED_DIR / "RF_model.tmp"
            ok = False
            if model_id:
                ok = _download_with_gdown(file_id=model_id, output=model_tmp) or _download_from_google_drive_fallback(model_id, model_tmp)
            if (not ok) and model_url:
                ok = _download_with_gdown(url=model_url, output=model_tmp) or _stream_download(model_url, model_tmp)
            if ok:
                model_tmp.replace(model_path)
                st.success("Model downloaded.")
            else:
                st.error("Failed to download model. Please set MODEL_URL or MODEL_DRIVE_ID (st.secrets or env).")
        else:
            st.error("No MODEL_URL or MODEL_DRIVE_ID provided; unable to retrieve the model.")

    # meta
    for p in meta_candidates:
        if p.exists() and p.stat().st_size > 0:
            meta_path = p
            break
    if meta_path is None:
        meta_url = _find_secret_or_env("META_URL")
        meta_id = _extract_drive_id(_find_secret_or_env("META_DRIVE_ID"))
        if meta_url or meta_id:
            st.info("Downloading model metadata…")
            tmp = TRAINED_DIR / "RF_model_meta.tmp.json"
            ok = False
            if meta_id:
                ok = _download_with_gdown(file_id=meta_id, output=tmp) or _download_from_google_drive_fallback(meta_id, tmp)
            if (not ok) and meta_url:
                ok = _download_with_gdown(url=meta_url, output=tmp) or _stream_download(meta_url, tmp)
            if ok:
                tmp.replace(TRAINED_DIR / "RF_model_meta.json")
                meta_path = TRAINED_DIR / "RF_model_meta.json"
                st.success("Metadata downloaded.")
            else:
                st.warning("Proceeding without metadata (optional).")

    return (model_path if model_path.exists() else None, meta_path)

# ==========================
# Loading
# ==========================

def _load_json_first(paths):
    for p in paths:
        try:
            if p and Path(p).exists():
                return json.loads(Path(p).read_text(encoding="utf-8")), Path(p)
        except Exception:
            continue
    return None, None

@st.cache_resource
def load_artifacts():
    model_p, _ = ensure_artifacts()
    if not model_p or not model_p.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    pipe = joblib.load(model_p)
    meta, meta_path = _load_json_first(META_PATHS)
    return pipe, meta, meta_path

def final_estimator(p):
    try:
        return p.steps[-1][1]
    except Exception:
        return p

def scroll_to_results():
    components.html(
        """
        <script>
        const el = window.parent.document.getElementById('results_top');
        if (el) { el.scrollIntoView({behavior: 'smooth', block: 'start'}); }
        </script>
        """,
        height=0,
    )

def get_column_transformer(pipeline):
    try:
        if hasattr(pipeline, "named_steps"):
            for key in ["prep", "preprocess", "preprocessor", "column_transformer"]:
                if key in pipeline.named_steps:
                    return pipeline.named_steps[key]
    except Exception:
        pass
    return None

def expected_columns(pipe, meta, csv_fallback: Path) -> list:
    if isinstance(meta, dict) and meta.get("feature_order"):
        return list(meta["feature_order"])

    try:
        ct = get_column_transformer(pipe)
        if ct is not None and hasattr(ct, "transformers_"):
            cols = []
            for _, _, selected in ct.transformers_:
                if isinstance(selected, (list, tuple, np.ndarray, pd.Index)):
                    cols.extend(list(selected))
            if cols:
                seen, ordered = set(), []
                for c in cols:
                    if c not in seen:
                        ordered.append(c)
                        seen.add(c)
                return ordered
    except Exception:
        pass

    for obj in (pipe, final_estimator(pipe)):
        try:
            if hasattr(obj, "feature_names_in_"):
                return list(obj.feature_names_in_)
        except Exception:
            pass

    try:
        if csv_fallback.exists():
            df = pd.read_csv(csv_fallback)
            drop = {"id", "obesity_class"}
            return [c for c in df.columns if c not in drop]
    except Exception:
        pass

    return [
        "Gender", "Age", "family_hist", "highcalorie", "vegtables", "main_meals", "snacks",
        "smokes", "water_intake", "monitors_calories", "physical_activity", "screen_time",
        "alcohol", "transport", "BMI",
    ]

def get_class_labels(pipe, meta):
    if isinstance(meta, dict) and meta.get("classes"):
        return [str(c) for c in meta["classes"]]
    try:
        fe = final_estimator(pipe)
        if hasattr(fe, "classes_"):
            return [str(c) for c in fe.classes_]
    except Exception:
        pass
    return None

def scale_to_normalized(value, original_min, original_max, normalized_min, normalized_max):
    if original_max == original_min:
        return float(normalized_min)
    return normalized_min + (value - original_min) * (normalized_max - normalized_min) / (
        original_max - original_min
    )

def bool_to_int(x: str) -> int:
    return 1 if str(x).strip().lower() in {"yes", "1", "true"} else 0

def never_to_zero(x: str) -> str:
    return "0" if str(x).strip().lower() == "never" else str(x)

def assemble_row(expected_cols: list, provided: dict) -> pd.DataFrame:
    if not expected_cols:
        return pd.DataFrame([{k: provided.get(k, np.nan) for k in provided.keys()}])
    row = {}
    for col in expected_cols:
        val = provided.get(col, np.nan)
        if col in {"snacks", "CAEC", "alcohol", "CALC"}:
            val = never_to_zero(val)
        row[col] = val
    return pd.DataFrame([row])

def render_probabilities(labels, probabilities):
    if labels is None or probabilities is None:
        return
    probs = list(zip(labels, probabilities))
    probs.sort(key=lambda x: x[1], reverse=True)
    for cls, p in probs:
        st.markdown(
            f"""
<div style="display:flex; align-items:center; justify-content:space-between; margin:.3rem 0 .2rem;">
  <div style="color:{PALETTE['text']}; font-weight:600;">{cls}</div>
  <div style="color:{PALETTE['text_muted']}; font-weight:700;">{p*100:.1f}%</div>
</div>
<div class="proba-bar"><div class="proba-fill" style="width:{p*100:.1f}%"></div></div>
""",
            unsafe_allow_html=True,
        )

# ==========================
# Load artifacts (with download if needed)
# ==========================
try:
    pipe, meta, meta_path = load_artifacts()
except Exception as e:
    st.error(f"Could not load prediction model. {e}")
    st.caption(f"Tried: {MODEL_PATH} (exists={MODEL_PATH.exists()})")
    st.stop()

EXPECTED_COLS = expected_columns(pipe, meta, CSV_FALLBACK)
CLASS_LABELS = get_class_labels(pipe, meta)

# ==========================
# UI
# ==========================

st.markdown(
    f"""
<div class="hero-card">
  <div style="display:flex; align-items:center; gap:.75rem;">
    <span class="badge" style="background:rgba(3,105,161,.07); color:{PALETTE['accent']}; border-color: rgba(3,105,161,.35)">Clinical tool</span>
    <span class="badge" style="background:rgba(15,118,110,.10); color:{PALETTE['primary_dim']}; border-color: rgba(15,118,110,.35)">Decision support</span>
  </div>
  <h1 style="color:{PALETTE['text']}; font-size:2.25rem; font-weight:900; margin:.35rem 0 0 0;">
    🏥 Health Analytics – Obesity Risk Assessment
  </h1>
  <p class="small" style="margin-top:.45rem;">
    Designed for clinical environments. This software supports assessment and does not replace clinical judgment.
  </p>
</div>
""",
    unsafe_allow_html=True,
)

left, right = st.columns([7, 5], gap="large")

with left:
    with st.form("input_form", clear_on_submit=False):
        st.markdown('<div class="section-header"><h3>Patient Details</h3></div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            gender = st.selectbox("Gender", ["Male", "Female"], help="Biological sex")
        with c2:
            age = st.number_input("Age (years)", min_value=14, max_value=100, value=25, step=1)
        with c3:
            height = st.number_input("Height (m)", min_value=1.0, max_value=2.5, value=1.70, step=0.01, format="%.2f")
        with c4:
            weight = st.number_input("Weight (kg)", min_value=30.0, max_value=250.0, value=70.0, step=0.5)

        try:
            bmi = float(round(weight / (height**2), 2))
        except Exception:
            bmi = 24.0

        st.markdown('<div class="section-header"><h3>Dietary & Family History</h3></div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            family_hist = st.selectbox("Family history of overweight", ["No", "Yes"])
        with c2:
            highcalorie = st.selectbox("High-calorie food intake", ["No", "Yes"])
        with c3:
            fcvc = st.slider("Vegetable servings/meal", 1, 6, 2, 1)
        with c4:
            ncp = st.slider("Main meals/day", 1, 6, 3, 1)

        st.markdown('<div class="section-header"><h3>Lifestyle & Activity</h3></div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            water_intake = st.slider("Water intake (L/day)", 1.0, 6.0, 2.0, 0.5)
            smokes = st.selectbox("Smoking status", ["No", "Yes"])
        with c2:
            faf = st.slider("Physical activity (hrs/day)", 0.0, 12.0, 1.0, 0.5)
            monitors_calories = st.selectbox("Monitors calories", ["No", "Yes"], help="Self-monitoring of energy intake")
        with c3:
            screen_time = st.slider("Screen time (hrs/day)", 0.0, 12.0, 1.0, 0.5)
            snacks = st.selectbox("Snacking frequency", ["Never", "Sometimes", "Frequently", "Always"])

        st.markdown('<div class="section-header"><h3>Transportation & Habits</h3></div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            alcohol = st.selectbox("Alcohol consumption", ["Never", "Sometimes", "Frequently"])
        with c2:
            transport = st.selectbox(
                "Primary transportation",
                ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"],
            )

        st.markdown("<div class='small'>Fields marked with standard units. Ensure values reflect typical patterns over the last 3 months.</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Run Assessment", type="primary")

with right:
    st.markdown("<div class='section-header'><h3>Results</h3></div>", unsafe_allow_html=True)
    st.markdown("<div id='results_top'></div>", unsafe_allow_html=True)
    placeholder = st.empty()

# Sidebar: model status
with st.sidebar:
    st.markdown("### Model Artifacts")
    st.caption(f"Model path: `{MODEL_PATH}`")
    exists = MODEL_PATH.exists()
    st.write("✅ Found" if exists else "❌ Missing")
    if exists:
        try:
            st.caption(f"Size: {MODEL_PATH.stat().st_size/1e6:.2f} MB")
        except Exception:
            pass
    if meta_path:
        st.caption(f"Meta: `{meta_path}`")

if submitted:
    # normalize / canonicalize
    fcvc_norm = scale_to_normalized(fcvc, 1, 6, 1.0, 3.0)
    ncp_norm = scale_to_normalized(ncp, 1, 6, 1.0, 3.0)
    water_norm = scale_to_normalized(water_intake, 1.0, 6.0, 1.0, 3.0)
    faf_norm = scale_to_normalized(faf, 0.0, 12.0, 0.0, 3.0)
    screen_norm = scale_to_normalized(screen_time, 0.0, 12.0, 0.0, 3.0)

    canonical_values = {
        "Gender": gender,
        "Age": int(age),
        "BMI": float(bmi),

        "family_hist": bool_to_int(family_hist),
        "highcalorie": bool_to_int(highcalorie),
        "vegtables": float(fcvc_norm),
        "main_meals": int(ncp_norm),
        "snacks": never_to_zero(snacks),
        "smokes": bool_to_int(smokes),
        "water_intake": float(water_norm),
        "monitors_calories": bool_to_int(monitors_calories),
        "physical_activity": float(faf_norm),
        "screen_time": float(screen_norm),
        "alcohol": never_to_zero(alcohol),
        "transport": transport,

        # Common alternate column names found in datasets
        "FAVC": bool_to_int(highcalorie),
        "FCVC": float(fcvc_norm),
        "NCP": int(ncp_norm),
        "CAEC": never_to_zero(snacks),
        "SMOKE": bool_to_int(smokes),
        "CH2O": float(water_norm),
        "SCC": bool_to_int(monitors_calories),
        "FAF": float(faf_norm),
        "TUE": float(screen_norm),

        "gender": gender,
        "age": int(age),
        "bmi": float(bmi),

        "Height": float(height),
        "Weight": float(weight),
        "height": float(height),
        "weight": float(weight),

        "water": float(water_norm),
        "tech_time": float(screen_norm),
    }

    X_row = assemble_row(EXPECTED_COLS, canonical_values)

    # Align to pipeline's feature order if available
    try:
        if hasattr(pipe, "feature_names_in_"):
            X_row = X_row.reindex(columns=list(pipe.feature_names_in_), fill_value=0)
    except Exception:
        pass

    try:
        if hasattr(pipe, "predict"):
            y_pred = pipe.predict(X_row)[0]
            label = str(y_pred)
        else:
            raise AttributeError("Loaded model does not implement .predict().")

        proba = None
        if hasattr(pipe, "predict_proba"):
            proba_arr = pipe.predict_proba(X_row)
            if isinstance(proba_arr, (list, np.ndarray)) and len(proba_arr) > 0:
                proba = proba_arr[0]

        with right:
            with placeholder.container():
                st.markdown(
                    f"""
<div class="result-card">
  <div class="badge">✓ Assessment complete</div>
  <div style="color:{PALETTE['text']}; font-size:2.1rem; font-weight:850; margin:.4rem 0 0 0;">
    {label}
  </div>
  <div class='small'>Based on patient inputs and model inference.</div>
</div>
""",
                    unsafe_allow_html=True,
                )

                st.markdown('<div class="stat-grid">', unsafe_allow_html=True)
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.markdown(
                        f"""
<div class="stat-card">
  <div class="k">BMI</div>
  <div class="v">{bmi:.2f}</div>
</div>
""",
                        unsafe_allow_html=True,
                    )
                with col_b:
                    st.markdown(
                        f"""
<div class="stat-card">
  <div class="k">Activity / day</div>
  <div class="v">{faf:.1f} h</div>
</div>
""",
                        unsafe_allow_html=True,
                    )
                with col_c:
                    st.markdown(
                        f"""
<div class="stat-card">
  <div class="k">Hydration</div>
  <div class="v">{water_intake:.1f} L</div>
</div>
""",
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)

                flags = []
                if bmi >= 30:
                    flags.append(("High BMI", PALETTE["danger"]))
                elif bmi >= 25:
                    flags.append(("Elevated BMI", PALETTE["warning"]))
                if faf < 1.0:
                    flags.append(("Very low activity", PALETTE["warning"]))
                if water_intake < 2.5:
                    flags.append(("Low hydration", PALETTE["warning"]))
                if screen_time > 6.0:
                    flags.append(("High screen time", PALETTE["warning"]))

                if flags:
                    st.markdown("### Attention areas")
                    for k, color in flags:
                        st.markdown(
                            f"<div class='custom-card' style='border-left:6px solid {color}'>🛈 {k}</div>",
                            unsafe_allow_html=True,
                        )
            scroll_to_results()

    except Exception as e:
        with right:
            st.error(f"Prediction Error: {e}")

st.markdown(
    f"""
<div class="app-footer">
  <div style="display:flex; align-items:center; justify-content:center; gap:.5rem; flex-wrap:wrap;">
    <span class="brand">Health Analytics</span>
    <span>•</span>
    <span>For professional use — not a substitute for medical advice. Additionally, please note that certain factors of this assessment will change your result depending on the inputs.</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)
