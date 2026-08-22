import os
import sys
import json
import requests
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import io

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Career & Salary Predictor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _get_secret(name, default=None):
    """Read a secret from Streamlit's secrets store, then env vars. Never
    surfaced in the UI — used only server-side so end users can't see which
    provider/model/key powers the assistant."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:  # noqa: BLE001 — no secrets.toml configured at all
        pass
    return os.environ.get(name, default)


ASSISTANT_API_KEY = _get_secret("GEMINI_API_KEY", "AQ.Ab8RN6Ky-V5y_FKjA88245QJ-AppNZmqcL4MJSwElgroP8d5qg")
ASSISTANT_MODEL_NAME = _get_secret("GEMINI_MODEL_NAME", "gemini-2.5-flash")


def semicolon_tokenizer(text):
    """'python;sql;machine learning' -> ['python', 'sql', 'machine learning']"""
    return text.split(";")


# Vectorizers were pickled with this tokenizer living in __main__ /
# this module, so both need to see it before joblib.load runs.
sys.modules["__main__"].semicolon_tokenizer = semicolon_tokenizer
sys.modules[__name__].semicolon_tokenizer = semicolon_tokenizer

# ---------------------------------------------------------------------------
# Model loading (unchanged logic, just cached + wrapped in a spinner)
# ---------------------------------------------------------------------------
os.makedirs("Models", exist_ok=True)

MODEL_URLS = {
    "linear_regression_salary_model.joblib": "https://drive.google.com/uc?export=download&id=1ae7FXIZluFdzzp0MLs6KOJS7PRFPn2EW",
    "career_recommendation_model.pkl": "https://drive.google.com/uc?export=download&id=1sz-1IytppzG4cZva5gBfyBoxONV29d7o",
    "career_label_encoder.pkl": "https://drive.google.com/uc?export=download&id=1SMwhSg3g9aCFR48eOZvLIFiuEBfLfvq0",
    "skills_vectorizer.pkl": "https://drive.google.com/uc?export=download&id=1ITP1paEQaTCla3SmV-RTliZCwjfXY8jG",
    "education_encoder.pkl": "https://drive.google.com/uc?export=download&id=1MqvOEGdeximjd-thffTcgrgcwRF40Yio",
    "interests_vectorizer.pkl": "https://drive.google.com/uc?export=download&id=1EaLpeIuwyp6uda6gE522lNYxH_KRb-wy",
}


def _download_drive_bytes(url, timeout=30):
    """GET a Google Drive 'uc?export=download' link and return the raw file
    bytes, robust to the interstitial HTML page Drive serves instead of the
    real file when it can't (or won't) stream a direct download — e.g. the
    "Google Drive can't scan this file for viruses" confirmation page, or a
    "download quota exceeded for this file" notice. Both cases return an
    HTTP 200 with an HTML body, so a plain requests.get() silently succeeds
    while actually downloading a warning page instead of your CSV/joblib —
    which is exactly what produced the empty-vocabulary error (the "CSV"
    being parsed had none of the expected columns).

    Raises RuntimeError with a clear, actionable message if Drive still
    won't hand over the real file after attempting the confirm-token flow.
    """
    session = requests.Session()
    resp = session.get(url, timeout=timeout, stream=True)
    resp.raise_for_status()

    def _is_html(r):
        return "text/html" in r.headers.get("Content-Type", "")

    if _is_html(resp):
        token = None
        # Older Drive flow: confirmation token comes back as a cookie.
        for k, v in resp.cookies.items():
            if k.startswith("download_warning"):
                token = v
                break
        # Newer Drive flow: token is embedded in the warning page itself.
        if token is None:
            m = re.search(r'confirm=([0-9A-Za-z_-]+)', resp.text)
            if m:
                token = m.group(1)
        if token is None:
            m = re.search(r'name="confirm"\s+value="([0-9A-Za-z_-]+)"', resp.text)
            if m:
                token = m.group(1)

        if token:
            sep = "&" if "?" in url else "?"
            resp = session.get(f"{url}{sep}confirm={token}", timeout=timeout, stream=True)
            resp.raise_for_status()

    if _is_html(resp):
        raise RuntimeError(
            "Google Drive returned a webpage instead of the file — this "
            "usually means the file's public-download quota was hit, or it "
            "needs a 'Download anyway' confirmation this app couldn't "
            "resolve automatically. Re-check the file's sharing settings "
            "(Anyone with the link) or try again shortly."
        )

    return resp.content


def download_if_missing(filename, url):
    dest = os.path.join("Models", filename)
    if not os.path.exists(dest):
        content = _download_drive_bytes(url, timeout=30)
        with open(dest, "wb") as f:
            f.write(content)
    return dest


@st.cache_resource(show_spinner="Loading models…")
def load_models():
    models = {}
    models["salary_model"] = joblib.load(
        download_if_missing("linear_regression_salary_model.joblib", MODEL_URLS["linear_regression_salary_model.joblib"])
    )
    models["career_recommendation_model"] = joblib.load(
        download_if_missing("career_recommendation_model.pkl", MODEL_URLS["career_recommendation_model.pkl"])
    )
    models["career_label_encoder"] = joblib.load(
        download_if_missing("career_label_encoder.pkl", MODEL_URLS["career_label_encoder.pkl"])
    )
    models["skills_vectorizer"] = joblib.load(
        download_if_missing("skills_vectorizer.pkl", MODEL_URLS["skills_vectorizer.pkl"])
    )
    models["education_encoder"] = joblib.load(
        download_if_missing("education_encoder.pkl", MODEL_URLS["education_encoder.pkl"])
    )
    models["interests_vectorizer"] = joblib.load(
        download_if_missing("interests_vectorizer.pkl", MODEL_URLS["interests_vectorizer.pkl"])
    )
    return models


try:
    Models = load_models()
    MODELS_OK = True
    MODELS_ERROR = None
except Exception as exc:  # noqa: BLE001
    Models = {}
    MODELS_OK = False
    MODELS_ERROR = str(exc)


def _clear_cached_data_files():
    """Delete the locally downloaded model/joblib/CSV copies and clear
    Streamlit's resource/data caches. download_if_missing() and the
    extended-career-index loader both skip re-downloading whenever a
    local copy already exists — which means updating a file on Google
    Drive (a new joblib, an extended CSV) has NO effect on an already-
    running deployment until the stale local copies are cleared. Call
    this (via the sidebar button below) after replacing any Drive file."""
    import shutil
    for folder in ("Models", "data"):
        shutil.rmtree(folder, ignore_errors=True)
    st.cache_resource.clear()
    st.cache_data.clear()

import re

EXTENDED_CSV_URL = "https://drive.google.com/uc?export=download&id=14EC3VJ3fRJJbEM41xo2V95Ul6kwzPUUS"

EXTENDED_INDEX_JOBLIB_URL = "https://drive.google.com/uc?export=download&id=1fJpraOlclBa0CQjhlvlOVcvOSYyQEgKv"
# Local cache of the built TF-IDF index, so it's only downloaded/rebuilt
# once per deployment instead of every session.
EXTENDED_INDEX_CACHE_PATH = os.path.join("Models", "extended_career_index.pkl")
CLEANED_CSV = os.path.join("data", "extended_careers_cleaned.csv")
LOCAL_RAW_CSV = os.path.join("data", "extended_careers.csv")


def clean_csv_inplace(src_path, dst_path):
    with open(src_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    # Heuristic 1: replace Windows newlines inside quotes with a space
    # This is conservative: only replace newline characters that are between quotes.
    text_fixed = re.sub(r'\"([^\"]*?)\n([^\"]*?)\"', lambda m: '"' + m.group(1).replace('\n', ' ') + m.group(2).replace('\n', ' ') + '"', text, flags=re.S)

    # Heuristic 2: collapse repeated quotes that look like broken escaping
    text_fixed = text_fixed.replace('""', '"')

    with open(dst_path, "w", encoding="utf-8") as out:
        out.write(text_fixed)


def load_joblib_from_drive(drive_url, cache_path=None, timeout=30):
    """Download a joblib/pkl file from Google Drive and return the loaded object.
    If cache_path is provided, save a local copy for future runs."""
    content = _download_drive_bytes(drive_url, timeout=timeout)
    obj = joblib.load(io.BytesIO(content))
    if cache_path:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            joblib.dump(obj, cache_path)
        except Exception:
            pass
    return obj


def _ensure_extended_columns(df):
    """Guarantee every column the rest of the app reads from the extended
    careers dataset exists, with a sane type/default — regardless of
    exactly which columns your extended CSV/joblib happens to include.
    Missing 'description' / 'avg_salary_inr' used to raise a KeyError as
    soon as a match was rendered; this makes a wider or reshaped dataset
    degrade gracefully instead of crashing the page."""
    for col in ["career", "skills", "interests", "description"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
    if "avg_salary_inr" not in df.columns:
        df["avg_salary_inr"] = 0.0
    df["avg_salary_inr"] = pd.to_numeric(df["avg_salary_inr"], errors="coerce").fillna(0.0)

    df["skills"] = df["skills"].str.replace(",", ";").str.replace(r"\s*;\s*", ";", regex=True)
    df["interests"] = df["interests"].str.replace(",", ";").str.replace(r"\s*;\s*", ";", regex=True)
    return df

def _valid_prebuilt_index(obj):
    """A prebuilt index is only usable if its vectorizer actually has a
    non-empty vocabulary. A previous bad download (Drive's HTML warning
    page saved and joblib-dumped as if it were real data) would otherwise
    get treated as "valid" forever since it satisfies the dict-shape check
    alone — this is what let the empty-vocabulary error persist across
    reruns via the local cache."""
    if not (isinstance(obj, dict) and {"df", "vectorizer", "matrix"}.issubset(obj.keys())):
        return False
    vec = obj.get("vectorizer")
    vocab = getattr(vec, "vocabulary_", None)
    return bool(vocab)


@st.cache_resource(show_spinner="Loading extended career dataset…")
def load_extended_career_index():
    """Load extended career index with these behaviors:
    1) If a local cached index exists at EXTENDED_INDEX_CACHE_PATH and is
       actually valid (non-empty vocabulary), use it.
    2) Else download the prebuilt index joblib from EXTENDED_INDEX_JOBLIB_URL.
    3) Else download the CSV from EXTENDED_CSV_URL and build the TF-IDF index.
    CSV parsing is tolerant: it will try to auto-detect delimiter and skip malformed lines.
    Both Drive downloads go through _download_drive_bytes(), which detects
    and works around the "Download anyway" / quota-exceeded HTML page Drive
    sometimes serves in place of the actual file — the root cause of the
    "empty vocabulary; perhaps the documents only contain stop words" error
    (an HTML warning page has no skills/interests columns, so every row's
    text ended up blank before it ever reached the vectorizer).
    The function always returns either an index dict {'df','vectorizer','matrix'} or None.
    """
    # 1) Try local cache first, but only trust it if it's genuinely usable.
    if os.path.exists(EXTENDED_INDEX_CACHE_PATH):
        try:
            cached = joblib.load(EXTENDED_INDEX_CACHE_PATH)
            if _valid_prebuilt_index(cached):
                cached["df"] = _ensure_extended_columns(cached["df"])
                return cached
        except Exception:
            pass

    # 2) Download the dedicated prebuilt index joblib (fast path — no
    # TF-IDF refitting needed).
    try:
        index = load_joblib_from_drive(EXTENDED_INDEX_JOBLIB_URL, cache_path=EXTENDED_INDEX_CACHE_PATH)
        if _valid_prebuilt_index(index):
            index["df"] = _ensure_extended_columns(index["df"])
            return index
        st.warning("The prebuilt career index joblib didn't contain a usable vocabulary — rebuilding from the CSV instead.")
    except Exception as exc:
        st.info(f"Couldn't load the prebuilt career index ({exc}); building it from the CSV instead.")

    # 3) Download CSV and build index (tolerant parsing). Downloaded to
    # disk and run through clean_csv_inplace first, since a CSV that's
    # been hand-edited in Sheets/Excel (e.g. to add more careers) commonly
    # picks up stray embedded newlines/quotes inside quoted fields.
    try:
        content = _download_drive_bytes(EXTENDED_CSV_URL, timeout=30)
        os.makedirs(os.path.dirname(LOCAL_RAW_CSV), exist_ok=True)
        with open(LOCAL_RAW_CSV, "wb") as raw_file:
            raw_file.write(content)
        clean_csv_inplace(LOCAL_RAW_CSV, CLEANED_CSV)
        with open(CLEANED_CSV, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

        # Try to detect delimiter using a small sample
        sample = "\n".join(text.splitlines()[:20])
        try:
            dialect = pd.io.common.csv.Sniffer().sniff(sample)
            delim = dialect.delimiter
        except Exception:
            if ";" in sample and sample.count(";") > sample.count(","):
                delim = ";"
            elif "\t" in sample and sample.count("\t") > sample.count(","):
                delim = "\t"
            else:
                delim = ","

        read_kwargs = {"sep": delim, "engine": "python", "dtype": str, "keep_default_na": False}
        try:
            df = pd.read_csv(io.StringIO(text), on_bad_lines="skip", **read_kwargs)
        except TypeError:
            df = pd.read_csv(io.StringIO(text), error_bad_lines=False, warn_bad_lines=False, **read_kwargs)

    except Exception as exc:
        st.warning(f"Couldn't load the extended careers dataset: {exc}")
        return None

    # Ensure every column the rest of the app relies on exists, with safe
    # defaults/types (covers a CSV that doesn't include every optional
    # column, e.g. "description" or "avg_salary_inr").
    df = _ensure_extended_columns(df)

    # Build TF-IDF index
    try:
        corpus = (df["skills"].str.replace(";", " ", regex=False) + " " + df["interests"].str.replace(";", " ", regex=False)).tolist()
        non_empty = [c for c in corpus if c.strip()]
        if not non_empty:
            # Give a diagnostic that actually points at the real cause,
            # instead of letting this fall through to TfidfVectorizer's
            # generic "empty vocabulary" ValueError.
            raise ValueError(
                f"no skills/interests text found in {len(df)} row(s) — got "
                f"columns {list(df.columns)}; the downloaded CSV likely "
                f"wasn't the real file (see the Drive download warning above, "
                f"if any) or the sharing link isn't set to 'Anyone with the link'."
            )
        vectorizer = TfidfVectorizer(lowercase=True)
        matrix = vectorizer.fit_transform(corpus)
        index = {"df": df, "vectorizer": vectorizer, "matrix": matrix}
    except Exception as exc:
        st.warning(f"Failed to build TF-IDF index from extended careers dataset: {exc}")
        return None

    # Cache locally for future runs (best-effort)
    try:
        os.makedirs(os.path.dirname(EXTENDED_INDEX_CACHE_PATH), exist_ok=True)
        joblib.dump(index, EXTENDED_INDEX_CACHE_PATH)
    except Exception:
        pass

    return index


def recommend_extended_careers(index, skills_selected, interests_selected, exclude_careers=(), top_n=4):
    """Fuzzy-match the user's skills/interests against the extended dataset
    and return the best new career suggestions not already covered by the
    trained classifier's own class list."""
    from sklearn.metrics.pairwise import cosine_similarity

    if not index or (not skills_selected and not interests_selected):
        return []

    query = " ".join(skills_selected) + " " + " ".join(interests_selected)
    query_vec = index["vectorizer"].transform([query])
    sims = cosine_similarity(query_vec, index["matrix"]).ravel()

    exclude_lower = {c.strip().lower() for c in exclude_careers}
    df = index["df"]
    ranked = sorted(
        (
            (df.iloc[i]["career"], float(sims[i]), df.iloc[i])
            for i in range(len(df))
            if sims[i] > 0 and df.iloc[i]["career"].strip().lower() not in exclude_lower
        ),
        key=lambda x: -x[1],
    )
    return ranked[:top_n]


EDUCATION_LEVELS = ["B.Tech", "High School", "M.Tech", "PhD"]
JOB_ROLES = ["Data Scientist", "Project Manager", "Software Engineer", "Data administrator"]
LOCATIONS = ["India", "UK", "USA", "Remote"] 
_EDU_CODE = {name: i for i, name in enumerate(EDUCATION_LEVELS)}
_ROLE_CODE = {name: i for i, name in enumerate(JOB_ROLES)}
_LOC_CODE = {name: i for i, name in enumerate(LOCATIONS)}

# ---------------------------------------------------------------------------
# Country average-salary index
# ---------------------------------------------------------------------------
# The regression model only ever saw four "Location" values during training
# (India / UK / USA / Remote). Previously, picking anything else silently
# extended the trained location code by +1 -- a meaningless extrapolation
# that didn't actually reflect how much more or less that country typically
# pays. Instead, every other country below is scaled off the model's
# India-based prediction using that country's approximate average annual
# salary (USD) relative to India's, so the overall salary *level* tracks
# real-world cost/pay differences while the model still supplies the shape
# of the curve (role, education, experience).
#
# Figures are broad, general-knowledge approximations of national average
# earnings -- NOT tech-salary-specific and NOT live data. They exist only to
# set a relative scale between countries. Swap in an authoritative source
# (World Bank, ILO, national statistics office) if you need this to be
# precise for a real deployment.
COUNTRY_INFO = {
    "India":         {"currency": "INR", "avg_salary_usd": 10000},
    "USA":           {"currency": "USD", "avg_salary_usd": 65000},
    "UK":            {"currency": "GBP", "avg_salary_usd": 45000},
    "Canada":        {"currency": "CAD", "avg_salary_usd": 55000},
    "Australia":     {"currency": "AUD", "avg_salary_usd": 60000},
    "New Zealand":   {"currency": "NZD", "avg_salary_usd": 50000},
    "Germany":       {"currency": "EUR", "avg_salary_usd": 55000},
    "France":        {"currency": "EUR", "avg_salary_usd": 48000},
    "Netherlands":   {"currency": "EUR", "avg_salary_usd": 58000},
    "Spain":         {"currency": "EUR", "avg_salary_usd": 32000},
    "Italy":         {"currency": "EUR", "avg_salary_usd": 34000},
    "Ireland":       {"currency": "EUR", "avg_salary_usd": 52000},
    "Switzerland":   {"currency": "CHF", "avg_salary_usd": 90000},
    "Sweden":        {"currency": "SEK", "avg_salary_usd": 48000},
    "Norway":        {"currency": "NOK", "avg_salary_usd": 62000},
    "Denmark":       {"currency": "DKK", "avg_salary_usd": 58000},
    "Finland":       {"currency": "EUR", "avg_salary_usd": 46000},
    "Poland":        {"currency": "PLN", "avg_salary_usd": 22000},
    "Russia":        {"currency": "RUB", "avg_salary_usd": 16000},
    "Turkey":        {"currency": "TRY", "avg_salary_usd": 13000},
    "Israel":        {"currency": "ILS", "avg_salary_usd": 48000},
    "UAE":           {"currency": "AED", "avg_salary_usd": 45000},
    "Saudi Arabia":  {"currency": "SAR", "avg_salary_usd": 32000},
    "Qatar":         {"currency": "QAR", "avg_salary_usd": 45000},
    "Japan":         {"currency": "JPY", "avg_salary_usd": 40000},
    "South Korea":   {"currency": "KRW", "avg_salary_usd": 38000},
    "China":         {"currency": "CNY", "avg_salary_usd": 18000},
    "Singapore":     {"currency": "SGD", "avg_salary_usd": 60000},
    "Malaysia":      {"currency": "MYR", "avg_salary_usd": 15000},
    "Thailand":      {"currency": "THB", "avg_salary_usd": 12000},
    "Indonesia":     {"currency": "IDR", "avg_salary_usd": 8500},
    "Philippines":   {"currency": "PHP", "avg_salary_usd": 9000},
    "Vietnam":       {"currency": "VND", "avg_salary_usd": 9500},
    "Pakistan":      {"currency": "PKR", "avg_salary_usd": 5500},
    "Bangladesh":    {"currency": "BDT", "avg_salary_usd": 5000},
    "Sri Lanka":     {"currency": "LKR", "avg_salary_usd": 6000},
    "Nepal":         {"currency": "NPR", "avg_salary_usd": 4500},
    "South Africa":  {"currency": "ZAR", "avg_salary_usd": 12000},
    "Nigeria":       {"currency": "NGN", "avg_salary_usd": 6000},
    "Kenya":         {"currency": "KES", "avg_salary_usd": 5000},
    "Egypt":         {"currency": "EGP", "avg_salary_usd": 6500},
    "Brazil":        {"currency": "BRL", "avg_salary_usd": 15000},
    "Mexico":        {"currency": "MXN", "avg_salary_usd": 14000},
    "Argentina":     {"currency": "ARS", "avg_salary_usd": 14000},
    "Chile":         {"currency": "CLP", "avg_salary_usd": 20000},
    "Colombia":      {"currency": "COP", "avg_salary_usd": 11000},
    "Remote":        {"currency": "USD", "avg_salary_usd": 45000},
}
ALL_COUNTRIES = sorted(COUNTRY_INFO.keys())


def country_relative_index(country, base_country="India"):
    """How much a country's average salary compares to the base country's
    (1.0 = same level, 2.0 = twice as high, 0.5 = half, etc.)."""
    base = COUNTRY_INFO.get(base_country, {}).get("avg_salary_usd", 10000)
    val = COUNTRY_INFO.get(country, {}).get("avg_salary_usd", base)
    return val / base if base else 1.0


def predict_salary_for_country(models, years_experience, education_level, job_role, country):
    """Predict a salary (in INR) for any country.

    For the four locations the regression model actually saw in training
    (India / UK / USA / Remote) this calls the model directly. For every
    other country, it takes the model's India-based prediction (which
    already reflects role, education, and experience) and rescales it by
    that country's average-salary index relative to India, since the model
    itself has no notion of, say, Germany or Nigeria.

    Returns (salary_inr, method) where method is "model" or "scaled" so the
    UI can be transparent about which one produced a given number.
    """
    if country in LOCATIONS:
        return predict_salary(models, years_experience, education_level, job_role, country), "model"
    base_inr = predict_salary(models, years_experience, education_level, job_role, "India")
    idx = country_relative_index(country, base_country="India")
    return base_inr * idx, "scaled"


# ---------------------------------------------------------------------------
# Job role pay-level index
# ---------------------------------------------------------------------------
# The regression model only ever saw four "JobRole" values during training
# (Data Scientist / Project Manager / Software Engineer / Data administrator).
# For every other role below, the same relative-scaling approach used for
# countries is applied: predict the anchor role ("Software Engineer") for
# the chosen location, then scale that number by the new role's approximate
# pay level relative to the anchor role, so the prediction reflects how
# that field typically pays rather than a meaningless extrapolated model
# category. Figures are broad, general-knowledge benchmarks (global,
# tech-skewed where relevant) -- NOT live data -- and exist only to set a
# relative scale between roles.
JOB_ROLE_ANCHOR = "Software Engineer"
JOB_ROLE_INFO = {
    # Roles the model was actually trained on
    "Software Engineer": 70000,
    "Data Scientist": 75000,
    "Project Manager": 68000,
    "Data administrator": 55000,
    # Tech / data
    "Machine Learning Engineer": 95000,
    "AI Researcher": 130000,
    "Data Analyst": 60000,
    "Data Engineer": 82000,
    "Database Administrator": 68000,
    "DevOps Engineer": 85000,
    "Site Reliability Engineer": 95000,
    "Cloud Architect": 110000,
    "Solutions Architect": 115000,
    "Full Stack Developer": 75000,
    "Backend Developer": 78000,
    "Frontend Developer": 68000,
    "Mobile App Developer": 72000,
    "Game Developer": 65000,
    "QA Engineer": 58000,
    "Systems Administrator": 60000,
    "Network Engineer": 65000,
    "Cybersecurity Analyst": 85000,
    "Security Engineer": 95000,
    "IT Support Specialist": 45000,
    "Technical Writer": 55000,
    "Research Scientist": 90000,
    "Statistician": 70000,
    "Blockchain Developer": 90000,
    # Design
    "UI/UX Designer": 65000,
    "Graphic Designer": 48000,
    "Product Designer": 80000,
    "Interior Designer": 50000,
    # Product / program / management
    "Product Manager": 95000,
    "Program Manager": 90000,
    "Scrum Master": 70000,
    "Operations Manager": 72000,
    "Engineering Manager": 120000,
    # Business / finance
    "Business Analyst": 62000,
    "Financial Analyst": 65000,
    "Accountant": 50000,
    "Investment Banker": 120000,
    "Supply Chain Analyst": 60000,
    "Consultant": 85000,
    "Entrepreneur": 60000,
    # People / marketing / sales
    "HR Manager": 65000,
    "Recruiter": 50000,
    "Marketing Manager": 70000,
    "Digital Marketing Specialist": 50000,
    "Content Writer": 42000,
    "Sales Executive": 55000,
    "Sales Manager": 80000,
    "Customer Success Manager": 65000,
    # Other professional fields
    "Legal Counsel": 100000,
    "Nurse": 55000,
    "Physician": 180000,
    "Pharmacist": 90000,
    "Teacher": 42000,
    "Professor": 65000,
    "Civil Engineer": 65000,
    "Mechanical Engineer": 70000,
    "Electrical Engineer": 72000,
    "Architect": 68000,
    "Journalist": 45000,
    "Photographer": 40000,
    "Video Editor": 45000,
    "Chef": 40000,
}
# Union of the model's own trained roles with the broader benchmark list
# above, so the UI can offer a much wider set of job roles than the model
# was originally trained on.
ALL_JOB_ROLES = sorted(set(JOB_ROLES) | set(JOB_ROLE_INFO.keys()))


def role_relative_index(job_role, base_role=JOB_ROLE_ANCHOR):
    """How much a role's typical pay compares to the anchor role's
    (1.0 = same level, 2.0 = twice as high, 0.5 = half, etc.)."""
    base = JOB_ROLE_INFO.get(base_role, 70000)
    val = JOB_ROLE_INFO.get(job_role, base)
    return val / base if base else 1.0


def predict_salary_full(models, years_experience, education_level, job_role, location):
    """Predict a salary (in INR) for ANY job role + location combination.

    - job_role is one of the model's four trained roles: delegates straight
      to predict_salary_for_country, which itself calls the model directly
      for a trained location, or location-scales it otherwise.
    - job_role is outside the trained set: predicts the anchor role
      ("Software Engineer") for the requested location (model or
      location-scaled, as above), then scales that number by job_role's
      pay-level index relative to the anchor role.

    Returns (salary_inr, method) where method is one of:
      "model"                    — trained role + trained location
      "scaled"                   — trained role, location scaled
      "role_scaled"              — role scaled, trained location
      "role_and_location_scaled" — both role and location scaled
    so the UI can explain exactly which extrapolation(s) produced a number.
    """
    if job_role in JOB_ROLES:
        return predict_salary_for_country(models, years_experience, education_level, job_role, location)

    base_inr, loc_method = predict_salary_for_country(
        models, years_experience, education_level, JOB_ROLE_ANCHOR, location
    )
    idx = role_relative_index(job_role)
    method = "role_and_location_scaled" if loc_method == "scaled" else "role_scaled"
    return base_inr * idx, method


# ---------------------------------------------------------------------------
# Expanded skill / interest tag library
# ---------------------------------------------------------------------------
# The trained vectorizers only recognize whatever vocabulary was present in
# the original training data. That's fine for the classifier itself (extra
# tags outside its vocabulary are simply ignored by that specific model),
# but it made the multiselect widgets feel very limited. These lists widen
# what's offered in the UI; they also feed the content-based "extended
# careers" matcher and the blended match score below, both of which pick up
# tags beyond the classifier's own fixed vocabulary.
EXTRA_SKILLS = [
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "kotlin", "swift", "php", "ruby", "r", "scala", "matlab", "sql", "nosql",
    "machine learning", "deep learning", "tensorflow", "pytorch", "keras",
    "nlp", "computer vision", "data analysis", "data visualization",
    "statistics", "statistical analysis", "pandas", "numpy", "spark",
    "hadoop", "tableau", "power bi", "excel", "a/b testing", "data mining",
    "big data", "etl", "react", "angular", "vue", "node.js", "html", "css",
    "rest api", "graphql", "django", "flask", "spring boot", "microservices",
    "system design", "software design", "git", "ci/cd", "testing", "agile",
    "scrum", "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
    "devops", "linux", "networking", "cybersecurity", "penetration testing",
    "cloud computing", "ui design", "ux design", "ui/ux", "figma",
    "adobe xd", "photoshop", "illustrator", "wireframing", "prototyping",
    "user research", "graphic design", "motion design", "project management",
    "product management", "business analysis", "communication", "leadership",
    "negotiation", "public speaking", "storytelling", "sales", "marketing",
    "seo", "content writing", "copywriting", "digital marketing",
    "social media marketing", "email marketing", "financial analysis",
    "accounting", "budgeting", "mobile app development", "android",
    "ios development", "game development", "unity", "blockchain",
    "robotics", "iot", "embedded systems", "quality assurance",
    "technical writing", "teaching", "research", "bioinformatics",
]
EXTRA_INTERESTS = [
    "academia", "ai", "analytics", "arts", "automation", "biotech",
    "business", "coding", "communications", "content", "cybersecurity",
    "data analysis", "data analytics", "data science", "design",
    "digital media", "electronics", "engineering", "finance", "gaming",
    "healthcare", "innovation", "language", "linguistics", "management",
    "marketing", "media", "mobile apps", "research", "security",
    "social media", "software development", "software engineering",
    "statistics", "technology", "user experience", "web design",
    "web development", "agriculture", "architecture", "aerospace",
    "astronomy", "environment", "sustainability", "climate", "education",
    "e-commerce", "entertainment", "fashion", "film", "food", "government",
    "policy", "hospitality", "law", "logistics", "manufacturing", "music",
    "nonprofit", "philanthropy", "photography", "psychology",
    "public health", "publishing", "real estate", "retail", "robotics",
    "space", "sports", "telecom", "tourism", "transportation",
    "venture capital", "entrepreneurship", "writing", "journalism",
]


def _merged_tag_vocab(models_vectorizer_key, extra_list):
    trained = (
        set(Models[models_vectorizer_key].vocabulary_.keys())
        if MODELS_OK and models_vectorizer_key in Models
        else set()
    )
    return sorted(trained | {t.lower() for t in extra_list})


ALL_SKILL_TAGS = _merged_tag_vocab("skills_vectorizer", EXTRA_SKILLS)
ALL_INTEREST_TAGS = _merged_tag_vocab("interests_vectorizer", EXTRA_INTERESTS)

# ---------------------------------------------------------------------------
# Prediction helpers (same underlying logic as the original script)
# ---------------------------------------------------------------------------


def predict_career(models, age, education, skills_str, interests_str):
    skills_list = [s.strip() for s in skills_str.split(";") if s.strip()]
    interests_list = [i.strip() for i in interests_str.split(";") if i.strip()]

    skills_vec = models["skills_vectorizer"].transform([";".join(skills_list)])
    interests_vec = models["interests_vectorizer"].transform([";".join(interests_list)])

    edu_encoded = models["education_encoder"].transform([[education]])
    edu_arr = edu_encoded.toarray() if hasattr(edu_encoded, "toarray") else np.asarray(edu_encoded)

    if edu_arr.size == 1:
        edu_sparse = csr_matrix([[float(edu_arr.ravel()[0])]])
    else:
        edu_sparse = csr_matrix(edu_arr)

    age_sparse = csr_matrix([[float(age)]])
    X_new_sparse = hstack([age_sparse, edu_sparse, skills_vec, interests_vec])

    try:
        X_new = X_new_sparse.toarray()
    except Exception:  # noqa: BLE001
        X_new = X_new_sparse

    model = models["career_recommendation_model"]
    classes = models["career_label_encoder"].classes_

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_new)[0]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_new)
        if scores.ndim == 1:
            scores = np.vstack([-scores, scores]).T
        exp = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probabilities = (exp / exp.sum(axis=1, keepdims=True))[0]
    else:
        pred = model.predict(X_new)
        probabilities = np.zeros(len(classes))
        probabilities[list(classes).index(pred[0])] = 1.0

    ranked = sorted(zip(classes, probabilities), key=lambda x: -x[1])
    return ranked[0][0], ranked


def _coverage_score(selected_set, required_tags_str):
    """What fraction of a career's typical skills/interests you've actually
    selected. Coverage-based (not similarity-based) on purpose: it only
    asks 'how much of what this career needs do you have', so it isn't
    diluted by extra, unrelated tags you also picked -- which is exactly
    the behavior that was missing before."""
    required = {t.strip().lower() for t in str(required_tags_str).split(";") if t.strip()}
    if not required:
        return 0.0
    return len(required & selected_set) / len(required)


def blended_career_matches(models, extended_index, age, education, skills_selected, interests_selected, top_n=5):
    """Score every career by how much of its typical skill/interest set you
    cover, then use the trained classifier's relative confidence to break
    ties among equally-covered careers. Returns [(career, match_score,
    raw_classifier_prob), ...] with match_score in [0, 0.97].

    Why coverage instead of raw classifier probability or cosine similarity:
    - The classifier has to spread a probability of 1.0 across *every*
      class it was trained on, so even a great match rarely tops 30-40%
      once there are a dozen-plus classes -- that's normal multi-class
      behavior, not a bug, but it reads as "broken" to a user.
    - Cosine similarity has the same shape problem (it's still a
      normalized share, not an absolute "how good is this match").
    - Coverage is absolute: if you've selected every skill and interest a
      career typically needs, its score approaches the cap regardless of
      what else you also selected or how many other careers exist. Select
      literally everything in the app, and — correctly — nearly every
      career will score very high, because by definition you now have
      every skill for every one of them; the classifier's relative
      confidence then decides the ordering among those top ties.
    - Capped at 97%, never 100%, because this is still a guidance tool,
      not a certainty claim.
    """
    skills_str = ";".join(skills_selected)
    interests_str = ";".join(interests_selected)
    skills_set = {s.strip().lower() for s in skills_selected}
    interests_set = {i.strip().lower() for i in interests_selected}

    clf_scores = {}
    if MODELS_OK:
        _, ranked = predict_career(models, age, education, skills_str, interests_str)
        clf_scores = {c: float(p) for c, p in ranked}
    # Rescale relative to the classifier's OWN top pick (so it can still
    # rank/break ties meaningfully) instead of using its raw, many-class-
    # diluted probability directly.
    max_clf = max(clf_scores.values()) if clf_scores else 0.0
    clf_rank_conf = {c: (p / max_clf if max_clf > 0 else 0.0) for c, p in clf_scores.items()}

    coverage_scores = {}
    if extended_index is not None and (skills_selected or interests_selected):
        df = extended_index["df"]
        for _, row in df.iterrows():
            career = str(row.get("career", "")).strip()
            if not career:
                continue
            skill_cov = _coverage_score(skills_set, row.get("skills", ""))
            interest_cov = _coverage_score(interests_set, row.get("interests", ""))
            score = 0.65 * skill_cov + 0.35 * interest_cov
            coverage_scores[career] = max(coverage_scores.get(career, 0.0), score)

    all_careers = set(clf_scores) | set(coverage_scores)
    if not all_careers:
        return []

    blended = {}
    for career in all_careers:
        coverage = coverage_scores.get(career)
        rank_conf = clf_rank_conf.get(career)
        if coverage is not None and rank_conf is not None:
            blended[career] = 0.75 * coverage + 0.25 * rank_conf
        elif coverage is not None:
            blended[career] = coverage
        else:
            # No coverage data for this career (extended dataset didn't
            # have it) -- fall back to a damped classifier signal so it
            # doesn't compete unfairly against coverage-backed scores.
            blended[career] = 0.5 * rank_conf

    ranked_blended = sorted(blended.items(), key=lambda x: -x[1])[:top_n]
    return [(career, min(score, 0.97), clf_scores.get(career, 0.0)) for career, score in ranked_blended]


def _code_for(value, code_map):
    """Look up the trained code for a value, or extend the mapping for a
    custom/'Other' value the model never saw during training. Custom
    values are an extrapolation and may be less reliable."""
    if value in code_map:
        return code_map[value]
    return max(code_map.values()) + 1


def predict_salary(models, years_experience, education_level, job_role, location):
    X_new = pd.DataFrame(
        {
            "YearsExperience": [years_experience],
            "EducationLevel": [_EDU_CODE.get(education_level, _code_for(education_level, _EDU_CODE))],
            "JobRole": [_code_for(job_role, _ROLE_CODE)],
            "Location": [_code_for(location, _LOC_CODE)],
        }
    )
    return float(models["salary_model"].predict(X_new)[0])


def salary_growth_curve(models, education_level, job_role, location, max_years=20, step=1):
    years = list(range(0, max_years + 1, step))
    salaries = [predict_salary_full(models, y, education_level, job_role, location)[0] for y in years]
    return years, salaries


def reference_average_salary(models, years_experience, education_level, location):
    """Average predicted salary across all trained job roles, used only to
    judge whether the user's specific prediction sits above/below that
    average — an internal reference point, not a claim about real market
    averages."""
    vals = [
        predict_salary_for_country(models, years_experience, education_level, role, location)[0]
        for role in JOB_ROLES
    ]
    return float(np.mean(vals))


# ---------------------------------------------------------------------------
# Currency handling — salary model output is treated as INR (the dataset's
# base currency), then converted for display based on the selected country.
# ---------------------------------------------------------------------------
CURRENCY_BY_LOCATION = {country: info["currency"] for country, info in COUNTRY_INFO.items()}
CURRENCY_SYMBOLS = {
    "INR": "₹", "USD": "$", "GBP": "£", "EUR": "€", "AUD": "A$", "CAD": "C$",
    "NZD": "NZ$", "CHF": "CHF ", "SEK": "kr", "NOK": "kr", "DKK": "kr",
    "PLN": "zł", "RUB": "₽", "TRY": "₺", "ILS": "₪", "AED": "AED ",
    "SAR": "SAR ", "QAR": "QAR ", "JPY": "¥", "KRW": "₩", "CNY": "¥",
    "SGD": "S$", "MYR": "RM", "THB": "฿", "IDR": "Rp", "PHP": "₱",
    "VND": "₫", "PKR": "₨", "BDT": "৳", "LKR": "Rs", "NPR": "Rs",
    "ZAR": "R", "NGN": "₦", "KES": "KSh", "EGP": "E£", "BRL": "R$",
    "MXN": "$", "ARS": "$", "CLP": "$", "COP": "$",
}
# Offline fallback (approximate, INR = 1 base unit). Used only if the live
# rate lookup fails or is unreachable — the live lookup (open.er-api.com)
# covers essentially all of these currencies under normal conditions.
FALLBACK_RATES_FROM_INR = {
    "INR": 1.0, "USD": 0.012, "GBP": 0.0095, "EUR": 0.011, "AUD": 0.018,
    "CAD": 0.0165, "NZD": 0.0196, "CHF": 0.0105, "SEK": 0.125, "NOK": 0.128,
    "DKK": 0.082, "PLN": 0.048, "RUB": 1.1, "TRY": 0.41, "ILS": 0.044,
    "AED": 0.044, "SAR": 0.045, "QAR": 0.0438, "JPY": 1.8, "KRW": 16.0,
    "CNY": 0.086, "SGD": 0.0161, "MYR": 0.056, "THB": 0.41, "IDR": 190.0,
    "PHP": 0.68, "VND": 300.0, "PKR": 3.36, "BDT": 1.4, "LKR": 3.6,
    "NPR": 1.6, "ZAR": 0.22, "NGN": 18.5, "KES": 1.56, "EGP": 0.59,
    "BRL": 0.071, "MXN": 0.21, "ARS": 12.8, "CLP": 11.9, "COP": 51.0,
}


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def get_exchange_rates():
    """Live INR -> other currency rates, with an offline fallback."""
    try:
        resp = requests.get("https://open.er-api.com/v6/latest/INR", timeout=4)
        resp.raise_for_status()
        data = resp.json()
        rates = data.get("rates", {})
        if rates:
            merged = dict(FALLBACK_RATES_FROM_INR)
            merged.update({k: v for k, v in rates.items() if k in CURRENCY_SYMBOLS})
            return merged, True
    except Exception:  # noqa: BLE001
        pass
    return dict(FALLBACK_RATES_FROM_INR), False


def convert_from_inr(amount_inr, target_currency, rates):
    return amount_inr * rates.get(target_currency, 1.0)


def format_money(amount, currency):
    symbol = CURRENCY_SYMBOLS.get(currency, currency + " ")
    if currency == "INR" and abs(amount) >= 100000:
        return f"{symbol} {amount / 100000:,.2f} L"
    if abs(amount) >= 1000:
        return f"{symbol}{amount:,.0f}"
    return f"{symbol}{amount:,.2f}"


# ---------------------------------------------------------------------------
# Styling — dark navy sidebar / light dashboard, per the reference mockup
# ---------------------------------------------------------------------------
CSS = """
<style>
:root {
    --navy: #0c1b3a;
    --navy-light: #142a56;
    --bg: #0a1226;
    --card: #131d38;
    --card-border: rgba(255,255,255,0.08);
    --text-dark: #eef1fb;
    --text-muted: #93a0c4;
    --accent-blue: #4d8dff;
    --accent-green: #16a34a;
    --accent-purple: #7c5cfc;
    --accent-orange: #f5921b;
    --accent-teal: #14b8c4;
    --accent-red: #dc2626;
}

/* ---- App shell ---------------------------------------------------- */
header[data-testid="stHeader"] { background: transparent; }
.stApp { background: var(--bg) !important; }

/* ---- Force readable, high-contrast text everywhere in the main
   content area, regardless of the viewer's light/dark browser theme.
   Scoped to .block-container so the sidebar (styled separately below)
   is untouched, and given !important so it always wins over Streamlit's
   own theme-driven defaults that caused the invisible-text bug. ------ */
/* Note: intentionally NOT touching bare <span> here — chips and skill
   pills use <span> with their own explicit accent colors below, and
   giving those a higher-specificity blanket override would clobber them. */
.block-container [data-testid="stMarkdownContainer"],
.block-container [data-testid="stMarkdownContainer"] p,
.block-container [data-testid="stMarkdownContainer"] li,
.block-container [data-testid="stMarkdownContainer"] h1,
.block-container [data-testid="stMarkdownContainer"] h2,
.block-container [data-testid="stMarkdownContainer"] h3,
.block-container [data-testid="stMarkdownContainer"] h4,
.block-container [data-testid="stMarkdownContainer"] h5,
.block-container [data-testid="stWidgetLabel"] p,
.block-container [data-testid="stWidgetLabel"] label {
    color: var(--text-dark) !important;
}
.block-container [data-testid="stCaptionContainer"],
.block-container [data-testid="stCaptionContainer"] * {
    color: var(--text-muted) !important;
}
.block-container h1, .block-container h2, .block-container h3,
.block-container h4, .block-container h5 { color: var(--text-dark) !important; }

/* Pin every input control to an explicit dark-navy pill style so it no
   longer depends on the visitor's OS light/dark preference — that
   mismatch (theme-driven text color vs. our forced-light card bg) was
   the root cause of text disappearing. */
.block-container [data-testid="stNumberInput"] input,
.block-container [data-testid="stTextInput"] input,
.block-container [data-baseweb="select"] > div,
.block-container [data-baseweb="base-input"] {
    background: var(--navy) !important;
    color: #f4f6fb !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}
.block-container [data-baseweb="select"] * ,
.block-container input,
.block-container textarea {
    color: #f4f6fb !important;
}
.block-container [data-testid="stNumberInput"] button svg { fill: #f4f6fb !important; }
.block-container [data-testid="stNumberInput"] button {
    background: var(--navy-light) !important;
    border-color: rgba(255,255,255,0.08) !important;
}
/* Multiselect chips (already-picked skills/interests) */
.block-container [data-baseweb="tag"] {
    background: var(--accent-blue) !important;
    color: #ffffff !important;
}
/* Dropdown option lists render in a portal outside .block-container.
   Pin that popover to the dark card style too, so it doesn't flash as
   a stray white box against the rest of the dark UI. */
div[data-baseweb="popover"], div[data-baseweb="menu"] {
    background: var(--card) !important;
}
div[data-baseweb="popover"] li, div[data-baseweb="menu"] li,
div[data-baseweb="popover"] li *, div[data-baseweb="menu"] li * {
    color: var(--text-dark) !important;
}
div[data-baseweb="popover"] li:hover, div[data-baseweb="menu"] li:hover {
    background: rgba(255,255,255,0.06) !important;
}

/* Chat bubbles (AI Career Assistant) — assistant on the left, user on
   the right, each as a rounded bubble instead of a plain full-width box. */
.block-container [data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 2px 0 !important;
    margin-bottom: 12px !important;
    display: flex;
    align-items: flex-start;
    gap: 10px;
}
.block-container [data-testid="stChatMessageContent"] {
    background: var(--card) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 16px !important;
    border-bottom-left-radius: 4px !important;
    padding: 10px 16px !important;
    max-width: 82%;
}
.block-container [data-testid="stChatMessageContent"] * {
    color: var(--text-dark) !important;
}
.block-container [data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) {
    flex-direction: row-reverse;
}
.block-container [data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) [data-testid="stChatMessageContent"] {
    background: rgba(77,141,255,0.18) !important;
    border-color: rgba(77,141,255,0.35) !important;
    border-radius: 16px !important;
    border-bottom-right-radius: 4px !important;
    border-bottom-left-radius: 16px !important;
}
.block-container [data-testid="stChatMessageAvatarUser"] {
    background: var(--accent-blue) !important;
}
.block-container [data-testid="stChatMessageAvatarAssistant"] {
    background: var(--accent-teal) !important;
}
.block-container [data-testid="stChatInput"] textarea {
    background: var(--navy) !important;
    color: #f4f6fb !important;
    border-radius: 10px !important;
}
.block-container [data-testid="stChatInput"] textarea::placeholder { color: #a9b4d4 !important; }

/* st.info / st.warning / st.error banners keep readable text on their own
   pastel background no matter the viewer's theme. */
.block-container [data-testid="stAlert"] * { color: var(--text-dark) !important; }

/* ---- Sidebar -------------------------------------------------------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--navy) 0%, var(--navy-light) 100%);
}
section[data-testid="stSidebar"] * { color: #e7ecff !important; }
section[data-testid="stSidebar"] .stRadio > label { display: none; }
section[data-testid="stSidebar"] div[role="radiogroup"] {
    display: flex;
    flex-direction: column;
    gap: 6px;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 0;
    transition: background 0.15s ease, border-color 0.15s ease;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: rgba(255,255,255,0.14);
}
/* Hide the visual radio dot entirely — the nav now reads as plain
   clickable pills, with the selected page shown via the highlighted
   background/left-bar below rather than a checked circle. Streamlit's
   radio renders the dot as the label's first child div (BaseWeb) with
   the native input hidden inside it for a11y — hiding that first child
   removes the dot while the label still forwards clicks to the input,
   so selection keeps working. */
section[data-testid="stSidebar"] div[role="radiogroup"] label[data-baseweb="radio"] > div:first-child,
section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
    display: none;
}
/* Highlight the whole pill (not just the dot) for the selected page,
   using :has() — supported in all current evergreen browsers. Falls
   back gracefully to the plain pill + recolored dot above if not. */
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(90deg, rgba(20,184,196,0.28), rgba(20,184,196,0.08));
    border-color: var(--accent-teal);
    box-shadow: inset 2px 0 0 var(--accent-teal);
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
    color: #ffffff !important;
    font-weight: 700;
}

/* ---- Header banner --------------------------------------------------- */
.app-header {
    background: linear-gradient(90deg, var(--navy) 0%, var(--navy-light) 100%);
    padding: 18px 26px;
    border-radius: 14px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 18px;
}
/* Note the extra .block-container [data-testid="stMarkdownContainer"]
   prefix here: without it, the earlier blanket "force every heading to
   --text-dark" rule above has equal-or-higher specificity and — since
   both use !important — silently wins by source order, leaving this
   heading a near-invisible dark-on-dark navy. Matching that prefix
   restores the intended white header text. */
.block-container [data-testid="stMarkdownContainer"] .app-header h1,
.app-header h1 { margin: 0; font-size: 1.5rem; color: #ffffff !important; }
.block-container [data-testid="stMarkdownContainer"] .app-header span.tag,
.app-header span.tag { color: #9db3ff !important; font-size: 0.85rem; }

/* ---- Cards ------------------------------------------------------------ */
.card {
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.35);
    margin-bottom: 16px;
}

/* ---- KPI chips ---------------------------------------------------------- */
.chip {
    border-radius: 12px;
    padding: 10px 14px;
    text-align: left;
    font-weight: 600;
}
.chip .label { display:block; font-size: 0.72rem; font-weight: 500; opacity: 0.85; }
.chip-blue,   .chip-blue   * { background: rgba(77,141,255,0.16);  color: #8fb6ff !important; }
.chip-green,  .chip-green  * { background: rgba(22,163,74,0.16);   color: #6ee0a0 !important; }
.chip-orange, .chip-orange * { background: rgba(245,146,27,0.18);  color: #fbbf6d !important; }
.chip-teal,   .chip-teal   * { background: rgba(20,184,196,0.18);  color: #7fe3ea !important; }

/* ---- Skill / requirement pills ------------------------------------------ */
.pill, .pill * {
    display: inline-block;
    background: rgba(255,255,255,0.07);
    color: #c3cbe6 !important;
    border-radius: 999px;
    padding: 4px 12px;
    margin: 3px 4px 3px 0;
    font-size: 0.82rem;
}
.pill.match, .pill.match * { background: rgba(22,163,74,0.22); color: #6ee0a0 !important; font-weight: 600; }

/* ---- Sidebar info boxes -------------------------------------------------- */
.side-box {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 10px 12px;
    margin-bottom: 10px;
    font-size: 0.85rem;
}
.side-box h4 { margin: 0 0 6px 0; font-size: 0.82rem; letter-spacing: .03em; text-transform: uppercase; color: #9db3ff !important; }
.side-box ul { margin: 0; padding-left: 18px; }
.disclaimer { border-left: 3px solid #f87171; }
.disclaimer h4 { color: #fca5a5 !important; }

/* ---- Buttons -------------------------------------------------------------- */
.block-container .stButton button {
    background: var(--accent-blue) !important;
    color: #ffffff !important;
    border: none !important;
    font-weight: 600;
    border-radius: 10px;
}
.block-container .stButton button:hover { background: #2559c9 !important; }

.footer-note { text-align:center; color: var(--text-muted) !important; font-size: 0.78rem; margin-top: 30px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🤖 Career & Salary AI")
    NAV_ICONS = {"Dashboard": "📊", "AI Career Assistant": "💬", "About System": "ℹ️"}
    page = st.radio(
        "Navigation",
        ["Dashboard", "AI Career Assistant", "About System"],
        label_visibility="collapsed",
        format_func=lambda x: f"{NAV_ICONS.get(x, '')}  {x}",
    )

    st.markdown(
        """
        <div class="side-box">
        <h4>Why This System</h4>
        <ul>
            <li>AI-powered salary prediction</li>
            <li>Personalized career suggestions</li>
            <li>Location-aware currency conversion</li>
            <li>Chat with an AI career assistant</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if MODELS_OK:
        demand_roles = list(Models["career_label_encoder"].classes_)[:6]
    else:
        demand_roles = JOB_ROLES
    roles_html = "".join(f"<li>{r}</li>" for r in demand_roles)

    # Live benchmark computed by actually running the salary model across
    # every trained job role (fixed baseline profile), instead of a static
    # hardcoded figure — so this panel moves whenever the model does.
    benchmark_html = ""
    if MODELS_OK:
        try:
            _baseline_years, _baseline_edu, _baseline_loc = 3, EDUCATION_LEVELS[0], "India"
            _role_salaries = {
                role: predict_salary(Models, _baseline_years, _baseline_edu, role, _baseline_loc)
                for role in JOB_ROLES
            }
            _top_role = max(_role_salaries, key=_role_salaries.get)
            _low_role = min(_role_salaries, key=_role_salaries.get)
            # Built as one unbroken line (no embedded newlines/indentation) —
            # when this got spliced into the outer indented f-string below,
            # its own indentation pushed it past Markdown's 4-space code-block
            # threshold, so it rendered as a raw <pre> block instead of HTML.
            benchmark_html = (
                f'<p style="margin:8px 0 2px 0;">📈 <b>Live benchmark</b> — '
                f'{_baseline_years} yrs exp, {_baseline_edu}, {_baseline_loc}:</p>'
                f'<ul style="margin:0;">'
                f'<li>Highest predicted: <b>{_top_role}</b> '
                f'({format_money(_role_salaries[_top_role], "INR")})</li>'
                f'<li>Lowest predicted: <b>{_low_role}</b> '
                f'({format_money(_role_salaries[_low_role], "INR")})</li>'
                f'</ul>'
            )
        except Exception:  # noqa: BLE001 — sidebar insight is best-effort
            benchmark_html = ""

    st.markdown(
        """
        <div class="side-box disclaimer">
        <h4>Disclaimer</h4>
        <p style="margin:0;">Estimates come from models trained on historical
        sample data — treat them as a guidance tool, not exact figures.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("🔄 Refresh data & models from Drive"):
        _clear_cached_data_files()
        st.rerun()

if not MODELS_OK:
    st.error(
        "Couldn't load the prediction models, so salary/career predictions "
        f"are unavailable right now.\n\nDetails: {MODELS_ERROR}"
    )

# ---------------------------------------------------------------------------
# Shared header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <div><h1>🤖 AI Based Salary Estimation &amp; Career Recommendation</h1></div>
        <div><span class="tag">Smart Career. Better Future.</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ===========================================================================
# DASHBOARD PAGE
# ===========================================================================
if page == "Dashboard":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Salary Estimation & Career Recommendation")
    st.caption("Enter your details to get an AI-predicted salary range and career matches.")

    education_options = (
        list(Models["education_encoder"].categories_[0]) if MODELS_OK else EDUCATION_LEVELS
    )
    skill_vocab = ALL_SKILL_TAGS
    interest_vocab = ALL_INTEREST_TAGS

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=15, max_value=80, value=25, step=1)
        education = st.selectbox("Education Level", education_options)
        job_role_choice = st.selectbox(
            "Job Role", ALL_JOB_ROLES + ["Other"], index=ALL_JOB_ROLES.index("Software Engineer")
        )
        if job_role_choice == "Other":
            job_role = st.text_input("Type the job role", key="job_role_other")
        else:
            job_role = job_role_choice
    with col2:
        years = st.number_input("Experience (Years)", min_value=0.0, max_value=50.0, value=2.0, step=0.5)
        location_choice = st.selectbox(
            "Location", ALL_COUNTRIES + ["Other"], index=ALL_COUNTRIES.index("India")
        )
        if location_choice == "Other":
            location = st.text_input("Type the location", key="location_other")
            display_currency = st.selectbox("Display currency", list(CURRENCY_SYMBOLS.keys()), index=1)
        else:
            location = location_choice
            display_currency = CURRENCY_BY_LOCATION.get(location, "USD")
            st.caption(f"Currency: **{display_currency}** (auto, based on location)")
    skills_selected = st.multiselect(
        "Skills", skill_vocab, help="Pick the skills that best describe you — a focused, realistic set "
        "gives sharper career matches than selecting everything."
    )
    interests_selected = st.multiselect("Interests", interest_vocab)

    if (job_role_choice == "Other") or (location_choice == "Other"):
        st.caption(
            "⚠️ 'Other' entries may be less reliable."
        )

    predict_clicked = st.button("Predict Salary & Recommend Careers", type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

    if predict_clicked and MODELS_OK:
        if job_role_choice == "Other" and not job_role.strip():
            st.warning("Type a job role.")
        elif location_choice == "Other" and not location.strip():
            st.warning("Type a location.")
        elif not skills_selected or not interests_selected:
            st.warning("Pick at least one skill and one interest for the career recommendation.")
        else:
            salary_inr, salary_method = predict_salary_full(Models, years, education, job_role, location)
            avg_reference_inr = reference_average_salary(Models, years, education, location)

            extended_index = load_extended_career_index()
            blended_matches = blended_career_matches(
                Models, extended_index, age, education, skills_selected, interests_selected, top_n=5
            )
            top_career = blended_matches[0][0] if blended_matches else "N/A"

            rates, live_rates = get_exchange_rates()
            salary_disp = convert_from_inr(salary_inr, display_currency, rates)
            min_disp = convert_from_inr(salary_inr * 0.85, display_currency, rates)
            max_disp = convert_from_inr(salary_inr * 1.15, display_currency, rates)
            avg_ref_disp = convert_from_inr(avg_reference_inr, display_currency, rates)

            st.session_state["last_prediction"] = {
                "age": age, "education": education, "years": years,
                "job_role": job_role, "location": location,
                "skills": skills_selected, "interests": interests_selected,
                "salary_inr": salary_inr, "salary_method": salary_method,
                "currency": display_currency,
                "salary_display": salary_disp, "top_career": top_career,
                "top_matches": blended_matches,
            }

            # ---- KPI chips row -------------------------------------------------
            c1, c2, c3, c4 = st.columns(4)
            for col, cls, label, value in [
                (c1, "chip-blue", "Experience", f"{years:g} yrs"),
                (c2, "chip-green", "Job Role", job_role),
                (c3, "chip-orange", "Top Career Match", top_career),
                (c4, "chip-teal", "Location", location),
            ]:
                col.markdown(
                    f'<div class="chip {cls}"><span class="label">{label}</span>{value}</div>',
                    unsafe_allow_html=True,
                )

            st.write("")
            left, right = st.columns([1, 1])

            # ---- Predicted salary + gauge --------------------------------------
            with left:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f"##### Predicted Salary ({display_currency})")
                st.markdown(f"**Estimated range:** {format_money(min_disp, display_currency)} – {format_money(max_disp, display_currency)}")
                st.markdown(f"<h2 style='color:#60a5fa !important;margin:4px 0;'>{format_money(salary_disp, display_currency)}</h2>", unsafe_allow_html=True)

                position = "Above Average" if salary_inr >= avg_reference_inr else "Below Average"
                pos_color = "#4ade80" if position == "Above Average" else "#f87171"

                gauge_max = max(max_disp * 1.1, salary_disp * 1.1, 1)
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=salary_disp,
                    number={"prefix": CURRENCY_SYMBOLS.get(display_currency, ""), "valueformat": ",.0f", "font": {"color": "#eef1fb"}},
                    gauge={
                        "axis": {"range": [0, gauge_max], "tickcolor": "#93a0c4"},
                        "bar": {"color": "#4d8dff"},
                        "bgcolor": "rgba(0,0,0,0)",
                        "steps": [
                            {"range": [0, gauge_max * 0.33], "color": "#7f1d1d"},
                            {"range": [gauge_max * 0.33, gauge_max * 0.66], "color": "#78350f"},
                            {"range": [gauge_max * 0.66, gauge_max], "color": "#14532d"},
                        ],
                        "threshold": {"line": {"color": pos_color, "width": 4}, "value": salary_disp},
                    },
                ))
                fig.update_layout(
                    height=220, margin=dict(l=20, r=20, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#eef1fb"),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(f"Market position: **<span style='color:{pos_color} !important'>{position}</span>** (vs. average across trained job roles for this profile)", unsafe_allow_html=True)
                if salary_method == "scaled":
                    st.caption(
                        f"💱 {location} isn't one of the model's trained locations "
                        f"(India/UK/USA/Remote), so this is the India-based prediction "
                        f"scaled by {location}'s average-salary level (~"
                        f"{country_relative_index(location):.2f}× India's)."
                    )
                elif salary_method == "role_scaled":
                    st.caption(
                        f"🧭 {job_role} isn't one of the model's trained job roles "
                        f"({', '.join(JOB_ROLES)}), so this is the {location} prediction for "
                        f"{JOB_ROLE_ANCHOR} scaled by {job_role}'s typical pay level (~"
                        f"{role_relative_index(job_role):.2f}× {JOB_ROLE_ANCHOR})."
                    )
                elif salary_method == "role_and_location_scaled":
                    st.caption(
                        f"🧭💱 Both {job_role} and {location} are outside what the model was "
                        f"trained on, so this prediction chains two scalings: the India-based "
                        f"{JOB_ROLE_ANCHOR} estimate scaled by {location}'s average-salary level "
                        f"(~{country_relative_index(location):.2f}× India's) and then by "
                        f"{job_role}'s typical pay level (~{role_relative_index(job_role):.2f}× "
                        f"{JOB_ROLE_ANCHOR}). Treat this as a rougher estimate than a single scaling."
                    )
                if not live_rates:
                    st.caption("Using offline exchange-rate estimates (live FX lookup unavailable).")
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("##### Salary Range Visualization")
                bar_fig = go.Figure(go.Bar(
                    x=["Min", "Avg", "Max"],
                    y=[min_disp, salary_disp, max_disp],
                    marker_color=["#f87171", "#4d8dff", "#4ade80"],
                    text=[format_money(v, display_currency) for v in [min_disp, salary_disp, max_disp]],
                    textposition="outside",
                    textfont=dict(color="#eef1fb"),
                ))
                bar_fig.update_layout(
                    height=260, margin=dict(l=20, r=20, t=10, b=10), showlegend=False,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#eef1fb"),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
                )
                st.plotly_chart(bar_fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # ---- Career recommendations + skills -------------------------------
            with right:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("##### Top Career Recommendations")
                if blended_matches:
                    for career, match_score, clf_prob in blended_matches:
                        st.write(f"**{career}** — {match_score * 100:.0f}% match")
                        st.progress(min(max(float(match_score), 0.0), 1.0))
                    st.caption(
                        "Match % reflects how much of that career's typical skills/interests "
                        "you've selected (not diluted by unrelated tags you also picked), with "
                        "the trained classifier's confidence used to rank ties. Cover most of a "
                        "career's skillset and its match can genuinely reach 80-90%+."
                    )
                else:
                    st.info("No career matches found for this combination of skills/interests.")
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("##### Skill & Interest Tag Library")
                tags_html = "".join(
                    f'<span class="pill match">{s}</span>' if s in skills_selected else f'<span class="pill">{s}</span>'
                    for s in skill_vocab
                )
                st.markdown(tags_html, unsafe_allow_html=True)
                st.caption(
                    "Highlighted pills are the skills you selected. All of these feed the "
                    "broader career-matching dataset; only the subset the classifier was "
                    "originally trained on also shapes its raw probability directly."
                )
                st.markdown("</div>", unsafe_allow_html=True)

            # ---- Career growth trend --------------------------------------------
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("##### Career Growth Trend")
            g_years, g_salaries_inr = salary_growth_curve(Models, education, job_role, location, max_years=20, step=1)
            g_salaries_disp = [convert_from_inr(s, display_currency, rates) for s in g_salaries_inr]
            growth_fig = go.Figure(go.Scatter(
                x=g_years, y=g_salaries_disp, mode="lines+markers",
                line=dict(color="#4ade80", width=3), marker=dict(size=5, color="#4ade80"),
            ))
            growth_fig.update_layout(
                height=280, margin=dict(l=20, r=20, t=10, b=10),
                xaxis_title="Years of Experience",
                yaxis_title=f"Salary ({display_currency})",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#eef1fb"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
            )
            st.plotly_chart(growth_fig, use_container_width=True)
            st.caption(
                "Projected by holding education, job role, and location fixed and "
                "varying years of experience — a model projection, not a guarantee."
            )
            st.markdown("</div>", unsafe_allow_html=True)

            # ---- Extended, dataset-driven career suggestions --------------------
            # (extended_index already loaded above, for the blended match score)
            known_classes = list(Models["career_label_encoder"].classes_) if MODELS_OK else []
            extra_matches = recommend_extended_careers(
                extended_index, skills_selected, interests_selected,
                exclude_careers=known_classes, top_n=4,
            )
            if extra_matches:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("##### More Careers Worth Exploring")
                st.caption(
                    "Matched from a broader career dataset by skill/interest overlap — "
                    "roles outside the trained model's core class list."
                )
                for career, score, row in extra_matches:
                    salary_disp_extra = convert_from_inr(float(row["avg_salary_inr"]), display_currency, rates)
                    st.markdown(
                        f"**{career}** — {score * 100:.0f}% match · "
                        f"~{format_money(salary_disp_extra, display_currency)}"
                    )
                    st.caption(row["description"])
                    req_pills = "".join(
                        f'<span class="pill">{s.strip()}</span>' for s in str(row["skills"]).split(";")
                    )
                    st.markdown(req_pills, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

    elif predict_clicked and not MODELS_OK:
        st.error("Models aren't loaded, so a prediction can't be made right now.")

    # ---- Model sensitivity diagnostic ---------------------------------
    # Answers "are my predictions actually changing?" with real numbers
    # pulled straight from the loaded models — not simulated. Runs the
    # salary model across every trained Job Role x Location combo (fixed
    # profile) and the career model across a few very different
    # skill/interest sets, then flags whether the output is effectively
    # flat. This exists to separate two very different root causes for
    # "predictions never change": (a) the app not passing your inputs
    # through correctly, vs (b) the underlying trained model itself
    # having weak/near-zero sensitivity to those inputs (common with a
    # small sample training set) or the extended career dataset failing
    # to load so the recommender falls back to one dominant class.
    with st.expander("🔍 Why aren't my results changing? (model sensitivity check)"):
        st.caption(
            "This calls the real loaded models across a few different inputs so "
            "you can see, directly, how much (if at all) each input actually "
            "moves the prediction — nothing here is simulated."
        )
        if not MODELS_OK:
            st.warning("Models aren't loaded, so this check can't run.")
        else:
            diag_years, diag_edu = 3, education_options[0]
            salary_rows = []
            for r in JOB_ROLES:
                for loc in LOCATIONS:
                    s_val, s_method = predict_salary_full(Models, diag_years, diag_edu, r, loc)
                    salary_rows.append(
                        {"Job Role": r, "Location": loc, "Predicted Salary (INR)": round(s_val), "Method": s_method}
                    )
            salary_diag_df = pd.DataFrame(salary_rows)
            st.markdown(
                f"**Salary model** — fixed profile ({diag_years} yrs exp, {diag_edu}), "
                "every trained Job Role × Location:"
            )
            st.dataframe(salary_diag_df, use_container_width=True, hide_index=True)
            s_min = salary_diag_df["Predicted Salary (INR)"].min()
            s_max = salary_diag_df["Predicted Salary (INR)"].max()
            s_mean = salary_diag_df["Predicted Salary (INR)"].mean()
            if s_mean and (s_max - s_min) < s_mean * 0.02:
                st.error(
                    "The salary model itself barely reacts to Job Role or Location for this "
                    "profile (< 2% spread) — that points to the trained model having weak "
                    "coefficients on those columns, not to a bug in this app's code. It was "
                    "likely trained on too small/uniform a sample to learn those effects. "
                    "Retraining `linear_regression_salary_model.joblib` on a larger, more "
                    "varied dataset is the real fix."
                )
            else:
                st.success(
                    f"The salary model does respond to Job Role/Location for this profile "
                    f"(spread: {format_money(s_min, 'INR')} – {format_money(s_max, 'INR')})."
                )

            st.markdown("---")
            st.markdown("**Career model** — same age/education, three very different skill/interest sets:")
            sample_sets = [
                (["python", "machine learning", "statistics"], ["ai", "data science"]),
                (["graphic design", "figma", "ui design"], ["design", "arts"]),
                (["sales", "negotiation", "communication"], ["marketing", "business"]),
            ]
            extended_index_diag = load_extended_career_index()
            career_rows = []
            for skl, intr in sample_sets:
                matches = blended_career_matches(Models, extended_index_diag, 25, diag_edu, skl, intr, top_n=1)
                top = matches[0][0] if matches else "N/A"
                career_rows.append({"Skills": ", ".join(skl), "Interests": ", ".join(intr), "Top Career": top})
            career_diag_df = pd.DataFrame(career_rows)
            st.dataframe(career_diag_df, use_container_width=True, hide_index=True)
            if extended_index_diag is None:
                st.error(
                    "The extended career dataset (used to make the match score responsive to "
                    "your exact skills/interests) failed to load — see any warning higher up "
                    "the page. With it unavailable, the top recommendation falls back to the "
                    "trained classifier alone, which can look 'stuck' on one class if it was "
                    "trained on a small/imbalanced sample."
                )
            elif len({row["Top Career"] for row in career_rows}) == 1:
                st.error(
                    "The top career recommendation is identical across very different "
                    "skill/interest sets. With the extended dataset loaded, that points to "
                    "the trained classifier being dominated by one class — again, most "
                    "likely a small/imbalanced training set rather than an app bug."
                )
            else:
                st.success("The career model does respond to different skills/interests.")

# ===========================================================================
# AI CAREER ASSISTANT
# ===========================================================================
elif page == "AI Career Assistant":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("AI Career Assistant")
    st.caption(
        "Chat about careers, skills, or your predicted salary. Answers are "
        "grounded in this app's career and skills data."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    def build_system_context():
        ctx = ["You are a helpful career and salary advisor embedded in a Streamlit app."]
        if MODELS_OK:
            ctx.append(f"Trained job roles: {JOB_ROLES}")
            ctx.append(
                f"Other supported job roles (salary estimated by scaling the {JOB_ROLE_ANCHOR} "
                f"prediction with each role's relative pay index, not directly modeled): "
                f"{[r for r in ALL_JOB_ROLES if r not in JOB_ROLES]}"
            )
            ctx.append(f"Locations the salary model was actually trained on: {LOCATIONS}")
            ctx.append(
                f"Other supported countries (salary estimated by scaling the India-based "
                f"prediction with each country's average-salary index, not directly modeled): "
                f"{[c for c in ALL_COUNTRIES if c not in LOCATIONS]}"
            )
            ctx.append(f"Trained education levels: {EDUCATION_LEVELS}")
            ctx.append(f"Career classes the recommender can output: {list(Models['career_label_encoder'].classes_)}")
            ctx.append(f"Skills vocabulary the classifier was trained on: {sorted(Models['skills_vectorizer'].vocabulary_.keys())}")
            ctx.append(f"Interests vocabulary the classifier was trained on: {sorted(Models['interests_vectorizer'].vocabulary_.keys())}")
            ctx.append(
                f"Full skill/interest tag library offered in the app UI (broader than the "
                f"classifier's own vocabulary; also used for content-based career matching): "
                f"skills={ALL_SKILL_TAGS}, interests={ALL_INTEREST_TAGS}"
            )
        last = st.session_state.get("last_prediction")
        if last:
            ctx.append(f"The user's most recent in-app prediction: {json.dumps(last, default=str)}")
        ctx.append(
            "Answer using this context when relevant. Be concise, practical, and honest "
            "about the limits of a model trained on a small sample dataset."
        )
        return "\n\n".join(ctx)

    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if not ASSISTANT_API_KEY:
        st.info(
            "The AI assistant isn't configured yet. Ask the site owner to set "
            "it up before chatting here."
        )
    else:
        user_msg = st.chat_input("Ask about careers, skills, or your predicted salary…")
        if user_msg:
            st.session_state["chat_messages"].append({"role": "user", "content": user_msg})
            with st.chat_message("user"):
                st.markdown(user_msg)

            with st.chat_message("assistant"):
                placeholder = st.empty()
                placeholder.markdown("Thinking…")
                try:
                    import google.generativeai as genai

                    genai.configure(api_key=ASSISTANT_API_KEY)
                    gmodel = genai.GenerativeModel(
                        model_name=ASSISTANT_MODEL_NAME,
                        system_instruction=build_system_context(),
                    )
                    history = [
                        {"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]}
                        for m in st.session_state["chat_messages"][:-1]
                    ]
                    chat = gmodel.start_chat(history=history)
                    response = chat.send_message(user_msg)
                    reply = response.text
                except Exception as exc:  # noqa: BLE001
                    # Never surface raw provider/key details to the end user —
                    # log the real error server-side only.
                    print(f"[AI Career Assistant] request failed: {exc}")
                    reply = "Sorry, I couldn't reach the assistant right now. Please try again in a moment."
                placeholder.markdown(reply)
            st.session_state["chat_messages"].append({"role": "assistant", "content": reply})

# ===========================================================================
# ABOUT PAGE
# ===========================================================================
else:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("About This System")
    st.markdown(
        f"""
- **Career recommendation** — blends two signals: the trained classifier (age, education,
  skills, interests → a fixed set of career classes) and a content-similarity score against
  a broader career dataset. Blending lets your full tag selection influence the result, since
  the classifier alone has to spread its confidence across every class it knows and stays
  low even for a strong match once there are many possible careers.
- **Skill & interest tags** — the app offers a wider tag library than the classifier's own
  training vocabulary; tags outside that vocabulary still feed the content-similarity matcher.
- **Currency conversion** — predictions are produced in INR, then converted to the currency
  of the selected country using a live exchange-rate lookup (falling back to fixed
  approximate rates if that lookup is unavailable).
- **AI Career Assistant** — a chat assistant grounded in the vocabulary/classes the
  models were trained on and your most recent in-app prediction.

This is a guidance tool built on a limited sample dataset and approximate national
salary and job-role pay figures — not a guarantee of real-world salary or career outcomes.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<p class="footer-note">Developed using Python, Machine Learning &amp; Streamlit</p>', unsafe_allow_html=True)
