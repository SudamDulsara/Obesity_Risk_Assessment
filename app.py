import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import hashlib, requests, os, tempfile, re


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
    "text": "#0f172a",
    "muted": "#475569",
    "primary": "#0ea5a4",
    "primary_dim": "#0891b2",
    "accent": "#14b8a6",
    "accent_dim": "#0ea5a4",
    "warn": "#eab308",
    "danger": "#ef4444",
    "ok": "#22c55e",
}

# ---------------------- styles ----------------------
st.markdown(
    f"""
<style>
html, body, [class*="css"]  {{
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, "Helvetica Neue", Arial;
}}

.main .block-container {{
  padding-top: 1.2rem; 
  padding-bottom: 4rem;
  max-width: 1200px;
}}

.app-header {{
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 1rem; gap: 1rem;
}}

.brand {{
  font-weight: 800; letter-spacing: .02em; 
  color: {PALETTE['text']};
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

.stat {{
  background: {PALETTE['card']};
  border: 1px solid {PALETTE['card_border']};
  border-radius: 14px;
  padding: .9rem;
}}

.stat .label {{
  color: {PALETTE['muted']};
  font-size: .8rem;
  font-weight: 700;
  letter-spacing: .03em;
  text-transform: uppercase;
}}

.stat .value {{
  font-size: 1.25rem;
  font-weight: 800;
  color: {PALETTE['text']};
}}

.panel {{
  background: {PALETTE['panel']};
  border: 1px solid {PALETTE['card_border']};
  border-radius: 16px;
  padding: 1rem;
}}

.section-title {{
  display:flex; align-items:center; justify-content:space-between; gap:.5rem;
  margin: .25rem 0 .8rem;
}}

.hrule {{
  height: 2px;
  background: linear-gradient(90deg, {PALETTE['primary']} 0%, {PALETTE['accent']} 100%);
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

.stNumberInput > div > div > input {{
  font-weight: 700;
}}

.small {{
  font-size: .85rem; color: {PALETTE['muted']};
}}

.app-footer {{
  margin-top: 2rem; padding-top: .75rem; 
  color: {PALETTE['muted']};
  border-top: 1px dashed {PALETTE['card_border']};
}}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------- Google Drive downloader (gdown + fallback) ----------------------
import re as _re_local
def _extract_drive_id(s: str) -> str | None:
    if not s:
        return None
    if _re_local.fullmatch(r"[A-Za-z0-9_-]{20,}", s):
        return s
    m = _re_local.search(r"/d/([A-Za-z0-9_-]{20,})", s) or _re_local.search(r"[?&]id=([A-Za-z0-9_-]{20,})", s)
    return m.group(1) if m else None

def _drive_uc_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={file_id}"

def _download_drive_with_requests(file_id: str, dest: Path):
    import requests as _requests_local
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    url = _drive_uc_url(file_id)
    with _requests_local.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    st.progress(min(downloaded / total, 1.0))
    tmp.replace(dest)

def _download_model(dest: Path):
    # Try Streamlit secrets first; if missing, fall back to env vars
    try:
        model_file_id = st.secrets.get("MODEL_FILE_ID")
        model_url = st.secrets.get("MODEL_URL", "")
    except Exception:
        # No secrets file in this environment
        model_file_id = os.getenv("MODEL_FILE_ID")
        model_url = os.getenv("MODEL_URL", "")

    file_id = model_file_id or _extract_drive_id(model_url)
    if not file_id:
        st.error(
            "Model location isn’t configured.\n\n"
            "Set **MODEL_FILE_ID** (or **MODEL_URL**) either in Streamlit Cloud → Settings → Secrets, "
            "or as environment variables. Example secrets:\n\n"
            'MODEL_FILE_ID = "1AbCDeFgHiJKlmNoPqrSTuvWXyz123456"\n'
        )
        st.stop()

    try:
        import gdown
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        gdown.download(id=file_id, output=str(tmp), quiet=False)
        tmp.replace(dest)
        return
    except ModuleNotFoundError:
        st.info("gdown not installed; using a direct download fallback. For large Drive files, add 'gdown==5.2.0' to requirements.txt.")
        _download_drive_with_requests(file_id, dest)


@st.cache_resource(show_spinner="Downloading model…")
def ensure_model_local(local_path: Path) -> Path:
    if local_path.exists():
        return local_path
    _download_model(local_path)
    if not local_path.exists() or local_path.stat().st_size == 0:
        st.error("Model download failed or produced an empty file.")
        st.stop()
    return local_path

# ---------------------- Paths (define MODEL_PATH before ensuring) ----------------------
APP_DIR = Path(__file__).resolve().parent
TRAINED_DIR = APP_DIR / "Trained"
PREPROC_DIR = APP_DIR / "Preprocessing" / "output"
CSV_FALLBACK = PREPROC_DIR / "cleaned_obesity_level.csv"
MODEL_PATH = TRAINED_DIR / "RF_model.pkl"
MODEL_PATH = ensure_model_local(MODEL_PATH)
META_PATHS = [TRAINED_DIR / "RF_model_meta.json", APP_DIR / "RF_model_meta.json"]

# ---------------------- (Optional) checksum helpers kept for future use ----------------------
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _stream_download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    st.progress(min(downloaded / total, 1.0))
        tmp.replace(dest)

# ---------------------- metadata utils ----------------------
def _load_json_first(paths):
    for p in paths:
        if Path(p).exists():
            with open(p, "r") as f:
                return json.load(f), str(p)
    return {}, None

# ---------------------- model/artifact loading ----------------------
@st.cache_resource
def load_artifacts(model_path: Path, meta_paths):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    pipe = joblib.load(model_path)
    meta, meta_path = _load_json_first(meta_paths)
    return pipe, meta, meta_path

# (removed duplicate load_artifacts)

# ---------------------- UI helpers ----------------------
def _hero():
    left, right = st.columns([0.7, 0.3])
    with left:
        st.markdown(
            f"""
<div class="app-header">
  <div>
    <div class="brand" style="font-size:1.35rem;">Obesity Risk Assessment</div>
    <div class="small">ML-aided screening tool — for educational & clinical research use only</div>
  </div>
  <div class="badge">🏥 Health Analytics</div>
</div>
<div class="hrule"></div>
""",
            unsafe_allow_html=True,
        )
    with right:
        st.metric("Model file", "RF_model.pkl", "loaded")

def _stat(label, value):
    st.markdown(
        f"""
<div class="stat">
  <div class="label">{label}</div>
  <div class="value">{value}</div>
</div>
""",
        unsafe_allow_html=True,
    )

def _input_number(lbl, min_value=0.0, max_value=1000.0, value=0.0, step=0.1):
    return st.number_input(lbl, min_value=min_value, max_value=max_value, value=value, step=step)

# ---------------------- main ----------------------
def main():
    _hero()

    pipe, meta, meta_path = load_artifacts(MODEL_PATH, META_PATHS)

    with st.expander("ℹ️ About this tool", expanded=False):
        st.write(
            "This app estimates obesity risk using a trained model. "
            "Results are for research/education and not a substitute for professional medical advice."
        )
        if meta_path:
            st.caption(f"Loaded metadata from: `{meta_path}`")

    st.markdown('<div class="section-title"><div class="brand">Patient Inputs</div></div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        age = _input_number("Age (years)", 1.0, 120.0, 30.0, 1.0)
        height = _input_number("Height (cm)", 50.0, 250.0, 170.0, 0.5)
        weight = _input_number("Weight (kg)", 10.0, 300.0, 70.0, 0.5)
    with c2:
        waist = _input_number("Waist Circumference (cm)", 30.0, 200.0, 80.0, 0.5)
        hip = _input_number("Hip Circumference (cm)", 30.0, 200.0, 90.0, 0.5)
        bp = _input_number("Systolic BP (mmHg)", 60.0, 220.0, 120.0, 1.0)
    with c3:
        chol = _input_number("Cholesterol (mg/dL)", 80.0, 350.0, 180.0, 1.0)
        hdl = _input_number("HDL (mg/dL)", 10.0, 120.0, 45.0, 1.0)
        trig = _input_number("Triglycerides (mg/dL)", 40.0, 600.0, 150.0, 1.0)

    # simple engineered features (placeholder – adjust to your pipeline’s expected features)
    height_m = height / 100.0
    bmi = weight / (height_m**2) if height_m > 0 else 0.0
    whr = waist / hip if hip > 0 else 0.0

    st.markdown('<div class="section-title"><div class="brand">Computed Features</div></div>', unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3)
    with g1:
        _stat("BMI", f"{bmi:.1f}")
    with g2:
        _stat("Waist-Hip Ratio", f"{whr:.2f}")
    with g3:
        _stat("Systolic BP", f"{bp:.0f} mmHg")

    # Construct model input row; adapt keys to your model’s training columns
    data = {
        "age": age,
        "height_cm": height,
        "weight_kg": weight,
        "waist_cm": waist,
        "hip_cm": hip,
        "systolic_bp": bp,
        "cholesterol": chol,
        "hdl": hdl,
        "triglycerides": trig,
        "bmi": bmi,
        "whr": whr,
    }
    X = pd.DataFrame([data])

    st.markdown('<div class="section-title"><div class="brand">Prediction</div></div>', unsafe_allow_html=True)
    if st.button("Predict Obesity Risk"):
        try:
            y_proba = None
            if hasattr(pipe, "predict_proba"):
                y_proba = pipe.predict_proba(X)
                risk = float(y_proba[0, 1]) if y_proba is not None and y_proba.ndim == 2 else None
            y_pred = pipe.predict(X)
            label = str(y_pred[0])

            if y_proba is not None:
                st.success(f"Predicted label: **{label}**  •  Estimated risk: **{risk:.2%}**")
            else:
                st.success(f"Predicted label: **{label}**")

        except Exception as e:
            st.error(f"Prediction Error: {e}")

    st.markdown(
        f"""
<div class="app-footer">
  <div style="display:flex; align-items:center; justify-content:center; gap:.5rem; flex-wrap:wrap;">
    <span class="brand">Health Analytics</span>
    <span>•</span>
    <span>For professional use — not a substitute for medical advice</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
