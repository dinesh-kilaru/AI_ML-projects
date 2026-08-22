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


def _resolve_assistant_api_key():
    """Prefer st.secrets / an env var over a key hard-coded in source.

    A real API key sitting in a .py file is a security problem on its own,
    but it's *also* the most likely reason the assistant is failing here:
    if this file (or a copy of it) was ever pushed to a public GitHub repo,
    Google's automated secret scanning typically finds and revokes keys
    like this within minutes to hours — which produces exactly this
    symptom (every call fails, no visible reason) with no code bug at all.

    If you're seeing this after pushing to GitHub, treat the key below as
    burned: generate a fresh one in Google AI Studio, revoke the old one,
    and put the new one in `.streamlit/secrets.toml` (as GEMINI_API_KEY) or
    in your host's environment variables — not back in this file.
    """
    try:
        key = st.secrets.get("GEMINI_API_KEY")
        if key:
            return key
    except Exception:
        pass
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    # Last-resort fallback so the app keeps working today; rotate this key.
    return "AIzaSyBFuGLVqKL8tNZnYX4a-RqZm9PfQQfeUXE"


ASSISTANT_API_KEY = _resolve_assistant_api_key()
# Fixed to a single model on purpose — no fallback to other Gemini models.
ASSISTANT_MODEL_CANDIDATES = ["gemini-2.5-flash"]


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


# ---------------------------------------------------------------------------
# Bundled, no-network fallback career dataset
# ---------------------------------------------------------------------------
# Root cause of "every profile recommends the same career": the ONLY source
# of skill/interest-based variety was the extended-careers dataset on
# Google Drive. Whenever that download failed (quota hit, bad sharing
# link, HTML warning page, etc. — see all the handling above), extended_index
# came back None, coverage_scores in blended_career_matches() was empty for
# every career, and the app silently fell back to the trained classifier
# alone — which, per its own class list, is easy to dominate with a single
# class on a small/imbalanced training sample. Same inputs in, same one
# career out, no matter what you picked.
#
# This table ships inside the app itself (no download, can't fail, can't go
# stale) and covers the full ALL_JOB_ROLES list. It's merged into whatever
# the Drive-based index does or doesn't provide, so skill/interest-driven
# variety no longer has a single external point of failure. Skill/interest
# tags are drawn from EXTRA_SKILLS / EXTRA_INTERESTS so they line up with
# what the multiselect widgets actually offer. Salary figures reuse the
# same broad JOB_ROLE_INFO USD benchmarks (roughly converted to INR) — they
# are approximations for relative ranking, not precise figures, consistent
# with every other disclaimer in this file.
STATIC_CAREER_PROFILES = {
    "Software Engineer": (["python", "java", "git", "system design", "testing", "agile"], ["software engineering", "coding", "technology"], "Builds and maintains software systems and applications."),
    "Data Scientist": (["python", "machine learning", "statistics", "pandas", "data analysis"], ["data science", "ai", "analytics"], "Extracts insights and builds predictive models from data."),
    "Project Manager": (["project management", "communication", "leadership", "agile"], ["management", "business"], "Plans and coordinates projects, timelines, and teams."),
    "Data administrator": (["sql", "nosql", "data analysis", "excel"], ["data analysis", "technology"], "Maintains and organizes an organization's databases."),
    "Machine Learning Engineer": (["python", "machine learning", "deep learning", "tensorflow", "pytorch"], ["ai", "data science", "technology"], "Designs and deploys machine learning systems in production."),
    "AI Researcher": (["machine learning", "deep learning", "research", "statistics"], ["ai", "research", "academia"], "Researches new methods and models in artificial intelligence."),
    "Data Analyst": (["sql", "excel", "data analysis", "tableau", "power bi"], ["data analytics", "analytics", "business"], "Analyzes data to answer business questions and build reports."),
    "Data Engineer": (["python", "sql", "spark", "hadoop", "etl", "big data"], ["data science", "technology", "engineering"], "Builds pipelines and infrastructure for large-scale data."),
    "Database Administrator": (["sql", "nosql", "linux", "networking"], ["technology", "data analysis"], "Manages, tunes, and secures production databases."),
    "DevOps Engineer": (["docker", "kubernetes", "ci/cd", "aws", "terraform", "linux"], ["software engineering", "technology", "automation"], "Automates deployment pipelines and infrastructure."),
    "Site Reliability Engineer": (["kubernetes", "docker", "linux", "aws", "devops"], ["technology", "engineering", "automation"], "Keeps large-scale systems reliable, observable, and fast."),
    "Cloud Architect": (["aws", "azure", "gcp", "cloud computing", "system design"], ["technology", "engineering"], "Designs cloud infrastructure and migration strategy."),
    "Solutions Architect": (["system design", "aws", "azure", "software design"], ["technology", "business", "engineering"], "Designs technical solutions that meet business requirements."),
    "Full Stack Developer": (["javascript", "react", "node.js", "html", "css", "sql"], ["web development", "coding", "software engineering"], "Builds both the front-end and back-end of web applications."),
    "Backend Developer": (["python", "java", "node.js", "sql", "rest api", "microservices"], ["software engineering", "web development"], "Builds server-side logic, APIs, and data layers."),
    "Frontend Developer": (["javascript", "react", "html", "css", "ui design"], ["web design", "user experience", "coding"], "Builds the user-facing interface of websites and apps."),
    "Mobile App Developer": (["swift", "kotlin", "mobile app development", "android", "ios development"], ["mobile apps", "technology", "coding"], "Builds native or cross-platform mobile applications."),
    "Game Developer": (["c++", "unity", "game development", "c#"], ["gaming", "technology", "entertainment"], "Builds gameplay systems and mechanics for video games."),
    "QA Engineer": (["testing", "quality assurance", "ci/cd", "agile"], ["software engineering", "technology"], "Tests software to find bugs before release."),
    "Systems Administrator": (["linux", "networking", "cybersecurity", "cloud computing"], ["technology", "security"], "Maintains servers, networks, and IT infrastructure."),
    "Network Engineer": (["networking", "linux", "cybersecurity", "cloud computing"], ["technology", "telecom"], "Designs and maintains computer networks."),
    "Cybersecurity Analyst": (["cybersecurity", "penetration testing", "networking", "linux"], ["security", "cybersecurity", "technology"], "Monitors and defends systems against security threats."),
    "Security Engineer": (["cybersecurity", "penetration testing", "cloud computing", "devops"], ["security", "cybersecurity", "engineering"], "Builds and hardens secure systems and infrastructure."),
    "IT Support Specialist": (["networking", "linux", "testing"], ["technology"], "Troubleshoots hardware, software, and network issues for users."),
    "Technical Writer": (["technical writing", "communication", "research"], ["writing", "technology", "content"], "Writes documentation, manuals, and technical guides."),
    "Research Scientist": (["research", "statistics", "data analysis"], ["research", "academia", "innovation"], "Conducts original research in a scientific or technical field."),
    "Statistician": (["statistics", "statistical analysis", "r", "data analysis"], ["statistics", "analytics", "research"], "Applies statistical methods to analyze and interpret data."),
    "Blockchain Developer": (["blockchain", "python", "javascript", "cybersecurity"], ["technology", "finance", "innovation"], "Builds decentralized applications and smart contracts."),
    "UI/UX Designer": (["ui design", "ux design", "figma", "wireframing", "prototyping"], ["design", "user experience", "arts"], "Designs how digital products look and feel to use."),
    "Graphic Designer": (["graphic design", "photoshop", "illustrator"], ["design", "arts", "media"], "Creates visual content for print and digital media."),
    "Product Designer": (["ui design", "ux design", "prototyping", "user research"], ["design", "user experience", "innovation"], "Designs end-to-end product experiences with user research."),
    "Interior Designer": (["graphic design", "prototyping"], ["design", "arts", "architecture"], "Designs functional and aesthetic interior spaces."),
    "Product Manager": (["product management", "communication", "leadership", "business analysis"], ["business", "management", "technology"], "Defines product strategy and coordinates its delivery."),
    "Program Manager": (["project management", "leadership", "communication"], ["management", "business"], "Coordinates multiple related projects toward shared goals."),
    "Scrum Master": (["agile", "scrum", "project management", "communication"], ["management", "software engineering"], "Facilitates agile teams and removes delivery blockers."),
    "Operations Manager": (["project management", "leadership", "budgeting"], ["management", "business", "logistics"], "Oversees daily operations and process efficiency."),
    "Engineering Manager": (["leadership", "system design", "project management", "communication"], ["engineering", "management", "technology"], "Leads engineering teams and technical direction."),
    "Business Analyst": (["business analysis", "data analysis", "excel", "communication"], ["business", "analytics"], "Bridges business needs and technical solutions."),
    "Financial Analyst": (["financial analysis", "excel", "statistics", "data analysis"], ["finance", "analytics", "business"], "Analyzes financial data to guide business decisions."),
    "Accountant": (["accounting", "budgeting", "excel"], ["finance", "business"], "Manages financial records, reporting, and compliance."),
    "Investment Banker": (["financial analysis", "negotiation", "excel"], ["finance", "business", "venture capital"], "Advises on capital raising, M&A, and financial deals."),
    "Supply Chain Analyst": (["data analysis", "excel", "project management"], ["logistics", "business", "manufacturing"], "Optimizes logistics, inventory, and supplier processes."),
    "Consultant": (["business analysis", "communication", "leadership", "negotiation"], ["business", "management", "innovation"], "Advises organizations on strategy and problem-solving."),
    "Entrepreneur": (["leadership", "negotiation", "communication", "project management"], ["entrepreneurship", "business", "innovation"], "Builds and runs a new business venture."),
    "HR Manager": (["communication", "leadership", "negotiation"], ["management", "business"], "Manages hiring, culture, and employee relations."),
    "Recruiter": (["communication", "negotiation", "public speaking"], ["business", "management"], "Sources and hires talent for an organization."),
    "Marketing Manager": (["marketing", "digital marketing", "seo", "communication"], ["marketing", "business", "media"], "Plans and leads marketing strategy and campaigns."),
    "Digital Marketing Specialist": (["digital marketing", "seo", "social media marketing", "email marketing"], ["marketing", "social media", "media"], "Runs online marketing campaigns across digital channels."),
    "Content Writer": (["content writing", "copywriting", "seo"], ["writing", "content", "media"], "Writes articles, copy, and content for audiences online."),
    "Sales Executive": (["sales", "negotiation", "communication"], ["business", "marketing"], "Sells products or services directly to customers."),
    "Sales Manager": (["sales", "leadership", "negotiation"], ["business", "management"], "Leads a sales team toward revenue targets."),
    "Customer Success Manager": (["communication", "negotiation", "project management"], ["business", "management"], "Helps customers get value from a product post-sale."),
    "Legal Counsel": (["negotiation", "communication", "research"], ["law", "business", "policy"], "Provides legal advice and manages compliance risk."),
    "Nurse": (["communication", "research"], ["healthcare", "public health"], "Provides direct patient care and support."),
    "Physician": (["research", "communication"], ["healthcare", "public health", "biotech"], "Diagnoses and treats patients as a medical doctor."),
    "Pharmacist": (["research", "quality assurance"], ["healthcare", "biotech", "public health"], "Dispenses medication and advises on drug safety."),
    "Teacher": (["teaching", "communication", "public speaking"], ["education", "academia"], "Teaches students in a school or classroom setting."),
    "Professor": (["teaching", "research", "public speaking"], ["academia", "education", "research"], "Teaches and researches at a college or university."),
    "Civil Engineer": (["system design", "project management"], ["engineering", "architecture", "sustainability"], "Designs and oversees construction of infrastructure."),
    "Mechanical Engineer": (["system design", "testing", "matlab"], ["engineering", "robotics", "manufacturing"], "Designs and tests mechanical systems and machines."),
    "Electrical Engineer": (["embedded systems", "iot", "matlab", "testing"], ["engineering", "electronics", "robotics"], "Designs electrical systems, circuits, and devices."),
    "Architect": (["prototyping", "system design"], ["architecture", "design", "arts"], "Designs buildings and physical spaces."),
    "Journalist": (["content writing", "communication", "research"], ["journalism", "writing", "media"], "Researches and reports news stories."),
    "Photographer": (["photoshop", "illustrator"], ["photography", "arts", "media"], "Captures and edits photographs professionally."),
    "Video Editor": (["motion design", "photoshop"], ["media", "film", "entertainment"], "Edits and produces video content."),
    "Chef": ([], ["food", "hospitality"], "Prepares and oversees the creation of food dishes."),
}


def _static_career_dataframe():
    rows = []
    for role, (skills, interests, desc) in STATIC_CAREER_PROFILES.items():
        rows.append({
            "career": role,
            "skills": ";".join(skills),
            "interests": ";".join(interests),
            "description": desc,
            # Rough USD->INR approximation of the JOB_ROLE_INFO benchmark,
            # for relative display only — see disclaimers near JOB_ROLE_INFO.
            "avg_salary_inr": JOB_ROLE_INFO.get(role, 60000) * 83.0,
        })
    return pd.DataFrame(rows)


def _build_tfidf_index(df):
    corpus = (df["skills"].str.replace(";", " ", regex=False) + " " + df["interests"].str.replace(";", " ", regex=False)).tolist()
    vectorizer = TfidfVectorizer(lowercase=True)
    matrix = vectorizer.fit_transform(corpus)
    return {"df": df, "vectorizer": vectorizer, "matrix": matrix}


def _merge_static_fallback(index):
    """Guarantee the returned index always has broad, varied skill/interest
    coverage, regardless of whether the Google Drive dataset loaded. If
    Drive succeeded, this only *adds* careers it didn't already cover; if
    Drive failed entirely, this becomes the whole dataset instead of None."""
    static_df = _ensure_extended_columns(_static_career_dataframe())
    if index is None or not _valid_prebuilt_index(index):
        return _build_tfidf_index(static_df)
    df = index["df"]
    have = set(df["career"].astype(str).str.strip().str.lower())
    extra = static_df[~static_df["career"].str.strip().str.lower().isin(have)]
    if extra.empty:
        return index
    merged_df = pd.concat([df, extra], ignore_index=True)
    return _build_tfidf_index(merged_df)


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
    A bundled, no-network fallback dataset (STATIC_CAREER_PROFILES) is always
    merged into the result via _merge_static_fallback(), so this function
    never returns None or a single-career-dominated dataset purely because
    a Google Drive download failed — see the comment above
    STATIC_CAREER_PROFILES for why that mattered.
    """
    # 1) Try local cache first, but only trust it if it's genuinely usable.
    if os.path.exists(EXTENDED_INDEX_CACHE_PATH):
        try:
            cached = joblib.load(EXTENDED_INDEX_CACHE_PATH)
            if _valid_prebuilt_index(cached):
                cached["df"] = _ensure_extended_columns(cached["df"])
                return _merge_static_fallback(cached)
        except Exception:
            pass

    # 2) Download the dedicated prebuilt index joblib (fast path — no
    # TF-IDF refitting needed).
    try:
        index = load_joblib_from_drive(EXTENDED_INDEX_JOBLIB_URL, cache_path=EXTENDED_INDEX_CACHE_PATH)
        if _valid_prebuilt_index(index):
            index["df"] = _ensure_extended_columns(index["df"])
            return _merge_static_fallback(index)
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
        return _merge_static_fallback(None)

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
        return _merge_static_fallback(None)

    index = _merge_static_fallback(index)

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
    base = COUNTRY_INFO.get(base_country, {}).get("avg_salary_usd", 10000)
    val = COUNTRY_INFO.get(country, {}).get("avg_salary_usd", base)
    return val / base if base else 1.0


def predict_salary_for_country(models, years_experience, education_level, job_role, country):
    if country in LOCATIONS:
        return predict_salary(models, years_experience, education_level, job_role, country), "model"
    base_inr = predict_salary(models, years_experience, education_level, job_role, "India")
    idx = country_relative_index(country, base_country="India")
    return base_inr * idx, "scaled"


JOB_ROLE_ANCHOR = "Software Engineer"
JOB_ROLE_INFO = {
    "Software Engineer": 70000,
    "Data Scientist": 75000,
    "Project Manager": 68000,
    "Data administrator": 55000,
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
    "UI/UX Designer": 65000,
    "Graphic Designer": 48000,
    "Product Designer": 80000,
    "Interior Designer": 50000,
    "Product Manager": 95000,
    "Program Manager": 90000,
    "Scrum Master": 70000,
    "Operations Manager": 72000,
    "Engineering Manager": 120000,
    "Business Analyst": 62000,
    "Financial Analyst": 65000,
    "Accountant": 50000,
    "Investment Banker": 120000,
    "Supply Chain Analyst": 60000,
    "Consultant": 85000,
    "Entrepreneur": 60000,
    "HR Manager": 65000,
    "Recruiter": 50000,
    "Marketing Manager": 70000,
    "Digital Marketing Specialist": 50000,
    "Content Writer": 42000,
    "Sales Executive": 55000,
    "Sales Manager": 80000,
    "Customer Success Manager": 65000,
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
ALL_JOB_ROLES = sorted(set(JOB_ROLES) | set(JOB_ROLE_INFO.keys()))


def role_relative_index(job_role, base_role=JOB_ROLE_ANCHOR):
    base = JOB_ROLE_INFO.get(base_role, 70000)
    val = JOB_ROLE_INFO.get(job_role, base)
    return val / base if base else 1.0


def predict_salary_full(models, years_experience, education_level, job_role, location):
    if job_role in JOB_ROLES:
        return predict_salary_for_country(models, years_experience, education_level, job_role, location)

    base_inr, loc_method = predict_salary_for_country(
        models, years_experience, education_level, JOB_ROLE_ANCHOR, location
    )
    idx = role_relative_index(job_role)
    method = "role_and_location_scaled" if loc_method == "scaled" else "role_scaled"
    return base_inr * idx, method


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
    required = {t.strip().lower() for t in str(required_tags_str).split(";") if t.strip()}
    if not required:
        return 0.0
    return len(required & selected_set) / len(required)


def blended_career_matches(models, extended_index, age, education, skills_selected, interests_selected, top_n=5):
    skills_str = ";".join(skills_selected)
    interests_str = ";".join(interests_selected)
    skills_set = {s.strip().lower() for s in skills_selected}
    interests_set = {i.strip().lower() for i in interests_selected}

    clf_scores = {}
    if MODELS_OK:
        _, ranked = predict_career(models, age, education, skills_str, interests_str)
        clf_scores = {c: float(p) for c, p in ranked}
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
            blended[career] = 0.5 * rank_conf

    ranked_blended = sorted(blended.items(), key=lambda x: -x[1])[:top_n]
    return [(career, min(score, 0.97), clf_scores.get(career, 0.0)) for career, score in ranked_blended]


def _code_for(value, code_map):
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
    vals = [
        predict_salary_for_country(models, years_experience, education_level, role, location)[0]
        for role in JOB_ROLES
    ]
    return float(np.mean(vals))


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
# Independent AI opinion (Dashboard) — a second, self-contained AI call
# ---------------------------------------------------------------------------
# Deliberately separate from everything above: it never touches Models,
# extended_index, salary_model.joblib, career_recommendation_model.pkl, or
# any other trained file, so it keeps working even if those fail to load.
# It DOES reuse ASSISTANT_API_KEY / ASSISTANT_MODEL_CANDIDATES exactly as
# already resolved above (no new key handling, nothing added/changed there),
# and it never shares state (history, session context) with the "AI Career
# Assistant" chat tab — that tab's code and behavior are untouched.
@st.cache_data(ttl=30 * 60, show_spinner=False)
def _fetch_independent_ai_opinion(age, education, years, job_role, location, skills_tuple, interests_tuple, currency):
    """Ask the AI for an independent salary/career opinion, using only its
    own general knowledge — no trained joblib model, no extended-careers
    dataset, and no shared chat history with the AI Career Assistant tab.
    Cached by input combination so repeat clicks with the same profile
    don't re-spend API quota. Returns (data_dict_or_None, error_or_None).
    """
    if not ASSISTANT_API_KEY:
        return None, "No API key is configured for the independent AI."
    try:
        import google.generativeai as genai
    except ImportError as exc:
        return None, (
            "The `google-generativeai` package isn't installed in this "
            f"environment.\n\n{exc}"
        )

    skills_list = list(skills_tuple)
    interests_list = list(interests_tuple)
    prompt = f"""You are an independent career and compensation analyst. You
have NO access to any specific proprietary dataset, trained model, or file
from this application — rely only on your own general knowledge of global
job markets, typical salary benchmarks, and career paths.

Candidate profile:
- Age: {age}
- Education level: {education}
- Years of experience: {years}
- Target job role: {job_role}
- Location: {location}
- Skills: {", ".join(skills_list) if skills_list else "none listed"}
- Interests: {", ".join(interests_list) if interests_list else "none listed"}

Respond with ONLY a raw JSON object (no markdown code fences, no extra
commentary before or after) with exactly these keys:
- "estimated_salary_range": a short string, a rough ANNUAL salary range in
  {currency}, formatted naturally for that currency (e.g. "₹6,00,000 - ₹9,50,000"
  or "$70,000 - $95,000")
- "reasoning": 1-2 concise sentences on how you arrived at that range
- "recommended_careers": a JSON array of up to 3 objects, each with a
  "career" string and a one-sentence "why" string
"""
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        gmodel = genai.GenerativeModel(model_name=ASSISTANT_MODEL_CANDIDATES[0])
        response = gmodel.generate_content(prompt)
        raw_text = (response.text or "").strip()
        cleaned = re.sub(r"^```(?:json)?|```$", "", raw_text, flags=re.MULTILINE).strip()
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            raise ValueError("response JSON wasn't an object")
        return data, None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def render_independent_ai_opinion(age, education, years, job_role, location, skills_selected, interests_selected, display_currency):
    """Render the dashboard's independent-AI card. Self-contained: reads
    nothing from Models/extended_index, and doesn't touch the AI Career
    Assistant tab's code, session state, or API-key resolution — it only
    reuses the already-resolved ASSISTANT_API_KEY/model name so the site
    owner still configures GEMINI_API_KEY in exactly one place."""
    st.markdown('<div class="ai-card">', unsafe_allow_html=True)
    st.markdown('<span class="ai-badge">Independent AI · not from trained files</span>', unsafe_allow_html=True)
    st.markdown("##### 🧠 Independent AI Salary & Career Opinion")
    st.caption(
        "A second opinion from a general-purpose AI reasoning from its own "
        "knowledge — separate from the joblib-trained salary/career models "
        "above, so it keeps working even if those model files fail to load."
    )

    if not ASSISTANT_API_KEY:
        st.info("This independent AI opinion isn't configured — no API key is available.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    with st.spinner("Asking the independent AI for its own estimate…"):
        data, error = _fetch_independent_ai_opinion(
            age, education, years, job_role, location,
            tuple(skills_selected), tuple(interests_selected), display_currency,
        )

    if error or not data:
        st.warning("Couldn't get an independent AI opinion right now.")
        with st.expander("Technical details (visible while debugging — remove for production)"):
            st.code(error or "Unknown error")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    salary_range = data.get("estimated_salary_range", "Not available")
    reasoning = data.get("reasoning", "")
    careers = data.get("recommended_careers", [])

    st.markdown(f"**Estimated annual salary range:** {salary_range}")
    if reasoning:
        st.caption(reasoning)

    if careers:
        st.markdown("**Careers this AI would suggest:**")
        for item in careers[:3]:
            if isinstance(item, dict):
                career = str(item.get("career", "")).strip()
                why = str(item.get("why", "")).strip()
                if career:
                    st.markdown(f"- **{career}** — {why}" if why else f"- **{career}**")

    st.caption(
        "Generated independently by a general-purpose AI model, not by this "
        "app's trained files — treat it as a second, rougher reference "
        "point rather than a precise figure."
    )
    st.markdown("</div>", unsafe_allow_html=True)


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

header[data-testid="stHeader"] { background: transparent; }
.stApp { background: var(--bg) !important; }

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
.block-container [data-baseweb="tag"] {
    background: var(--accent-blue) !important;
    color: #ffffff !important;
}
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

.block-container [data-testid="stAlert"] * { color: var(--text-dark) !important; }

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
section[data-testid="stSidebar"] div[role="radiogroup"] label[data-baseweb="radio"] > div:first-child,
section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
    display: none;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(90deg, rgba(20,184,196,0.28), rgba(20,184,196,0.08));
    border-color: var(--accent-teal);
    box-shadow: inset 2px 0 0 var(--accent-teal);
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
    color: #ffffff !important;
    font-weight: 700;
}

.app-header {
    background: linear-gradient(90deg, var(--navy) 0%, var(--navy-light) 100%);
    padding: 18px 26px;
    border-radius: 14px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 18px;
}
.block-container [data-testid="stMarkdownContainer"] .app-header h1,
.app-header h1 { margin: 0; font-size: 1.5rem; color: #ffffff !important; }
.block-container [data-testid="stMarkdownContainer"] .app-header span.tag,
.app-header span.tag { color: #9db3ff !important; font-size: 0.85rem; }

.card {
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.35);
    margin-bottom: 16px;
}

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

.block-container .stButton button {
    background: var(--accent-blue) !important;
    color: #ffffff !important;
    border: none !important;
    font-weight: 600;
    border-radius: 10px;
}
.block-container .stButton button:hover { background: #2559c9 !important; }

.footer-note { text-align:center; color: var(--text-muted) !important; font-size: 0.78rem; margin-top: 30px; }

.ai-card {
    background: linear-gradient(135deg, rgba(124,92,252,0.14), rgba(20,184,196,0.08));
    border: 1px solid rgba(124,92,252,0.35);
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.35);
    margin-bottom: 16px;
}
.ai-card h5, .ai-card p, .ai-card li { color: var(--text-dark) !important; }
.ai-badge {
    display: inline-block;
    background: rgba(124,92,252,0.28);
    color: #d7cbff !important;
    border-radius: 999px;
    padding: 3px 11px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: .03em;
    text-transform: uppercase;
    margin-bottom: 8px;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

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

st.markdown(
    """
    <div class="app-header">
        <div><h1>🤖 AI Based Salary Estimation &amp; Career Recommendation</h1></div>
        <div><span class="tag">Smart Career. Better Future.</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

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

            # -----------------------------------------------------------------
            # Independent AI opinion (Gemini) — deliberately separate from the
            # trained joblib models above. See _independent_ai_dashboard_opinion()
            # for details; it reuses the SAME API key resolution as the AI
            # Career Assistant tab (nothing about that tab or the key handling
            # is touched), but makes its own standalone Gemini call with no
            # shared chat history and no dependency on Models/extended_index.
            # -----------------------------------------------------------------
            render_independent_ai_opinion(
                age=age, education=education, years=years, job_role=job_role,
                location=location, skills_selected=skills_selected,
                interests_selected=interests_selected,
                display_currency=display_currency,
            )

    elif predict_clicked and not MODELS_OK:
        st.error("Models aren't loaded, so a prediction can't be made right now.")
        st.caption(
            "The trained-model prediction is unavailable, but you can still try "
            "the independent AI estimate below — it doesn't depend on the "
            "joblib model files at all."
        )
        render_independent_ai_opinion(
            age=age, education=education, years=years, job_role=job_role,
            location=location, skills_selected=skills_selected,
            interests_selected=interests_selected,
            display_currency=CURRENCY_BY_LOCATION.get(location, "USD"),
        )

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

            # load_extended_career_index() now always merges in the bundled,
            # no-network STATIC_CAREER_PROFILES fallback, so it can no longer
            # be None — but it's still worth telling you whether the Google
            # Drive dataset actually loaded, or whether you're running on the
            # bundled fallback alone (fewer careers, but never "stuck").
            static_careers = set(STATIC_CAREER_PROFILES.keys())
            diag_careers = set(extended_index_diag["df"]["career"].astype(str).str.strip()) if extended_index_diag else set()
            drive_loaded = bool(diag_careers - static_careers)
            if not drive_loaded:
                st.warning(
                    "The Google Drive extended-careers dataset didn't load (see any warning "
                    "higher up the page) — recommendations are currently running on the "
                    f"{len(static_careers)}-career fallback dataset bundled in the app itself, "
                    "instead of your full Drive dataset. Career variety still works (see the "
                    "table above), just from a smaller pool. Fix the Drive sharing/quota issue "
                    "to bring back your full dataset."
                )
            if len({row["Top Career"] for row in career_rows}) == 1:
                st.error(
                    "The top career recommendation is identical across very different "
                    "skill/interest sets even with career-matching data loaded. That points to "
                    "a bug in the matching logic itself, not just missing data — worth "
                    "reporting/investigating further."
                )
            else:
                st.success("The career model does respond to different skills/interests.")

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
                reply = None
                debug_detail = None
                try:
                    import google.generativeai as genai
                except ImportError as exc:
                    debug_detail = (
                        "ImportError: the `google-generativeai` package isn't installed in "
                        "this environment. Add `google-generativeai` to requirements.txt and "
                        f"redeploy.\n\n{exc}"
                    )
                    reply = "Sorry, I couldn't reach the assistant right now. Please try again in a moment."

                if reply is None:
                    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
                    history = [
                        {"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]}
                        for m in st.session_state["chat_messages"][:-1]
                    ]
                    system_ctx = build_system_context()
                    last_exc = None
                    # Cache whichever candidate model name actually worked, so
                    # later turns don't re-probe every candidate from scratch.
                    model_order = st.session_state.get(
                        "_working_gemini_model", ASSISTANT_MODEL_CANDIDATES
                    )
                    if isinstance(model_order, str):
                        model_order = [model_order] + [m for m in ASSISTANT_MODEL_CANDIDATES if m != model_order]
                    for candidate in model_order:
                        try:
                            gmodel = genai.GenerativeModel(
                                model_name=candidate,
                                system_instruction=system_ctx,
                            )
                            chat = gmodel.start_chat(history=history)
                            response = chat.send_message(user_msg)
                            reply = response.text
                            st.session_state["_working_gemini_model"] = candidate
                            break
                        except Exception as exc:  # noqa: BLE001
                            last_exc = exc
                            continue

                    if reply is None:
                        # Log the real error server-side, and translate the
                        # most common failure signatures into an actionable
                        # hint instead of a generic dead end.
                        print(f"[AI Career Assistant] request failed: {last_exc}")
                        msg = str(last_exc)
                        if any(t in msg for t in ("API_KEY_INVALID", "API key not valid", "401", "PERMISSION_DENIED", "403")):
                            debug_detail = (
                                "Auth error: the Gemini API key is invalid, revoked, or lacks "
                                "access to this model. If this key was ever committed to a "
                                "public repo, Google likely auto-revoked it — generate a new "
                                "key in Google AI Studio and store it as GEMINI_API_KEY in "
                                f"Streamlit secrets.\n\n{msg}"
                            )
                        elif any(t in msg for t in ("404", "not found", "NOT_FOUND")):
                            debug_detail = (
                                "Model-not-found error: none of "
                                f"{ASSISTANT_MODEL_CANDIDATES} are available to this API key/"
                                f"project. Check available model names for your account in "
                                f"Google AI Studio.\n\n{msg}"
                            )
                        elif any(t in msg for t in ("429", "quota", "RESOURCE_EXHAUSTED")):
                            debug_detail = f"Rate-limit/quota error from the Gemini API.\n\n{msg}"
                        else:
                            debug_detail = f"Unhandled error calling the Gemini API.\n\n{msg}"
                        reply = "Sorry, I couldn't reach the assistant right now. Please try again in a moment."

                placeholder.markdown(reply)
                if debug_detail:
                    with st.expander("Technical details (visible while debugging — remove for production)"):
                        st.code(debug_detail)
            st.session_state["chat_messages"].append({"role": "assistant", "content": reply})

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
- **Independent AI opinion (Dashboard)** — after a prediction, a separate Gemini call
  (no shared context with the chat assistant, no dependency on the joblib model files)
  gives its own salary-range and career-fit opinion from general knowledge alone, so you
  have a second, model-free reference point to compare against the trained-model output.

This is a guidance tool built on a limited sample dataset and approximate national
salary and job-role pay figures — not a guarantee of real-world salary or career outcomes.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<p class="footer-note">Developed using Python, Machine Learning &amp; Streamlit</p>', unsafe_allow_html=True)
