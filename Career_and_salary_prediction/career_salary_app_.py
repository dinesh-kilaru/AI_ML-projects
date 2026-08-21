
import sys
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from scipy.sparse import hstack
import os
import requests
import joblib
import streamlit as st

def semicolon_tokenizer(text):
    """'python;sql;machine learning' -> ['python', 'sql', 'machine learning']"""
    return text.split(";")

# Ensure Models directory exists
os.makedirs("Models", exist_ok=True)

# Map each model to its Google Drive direct download link
MODEL_URLS = {
    "linear_regression_salary_model.joblib": "https://drive.google.com/uc?export=download&id=1ae7FXIZluFdzzp0MLs6KOJS7PRFPn2EW",
    "career_recommendation_model.pkl": "https://drive.google.com/uc?export=download&id=1sz-1IytppzG4cZva5gBfyBoxONV29d7o",
    "career_label_encoder.pkl": "https://drive.google.com/uc?export=download&id=1SMwhSg3g9aCFR48eOZvLIFiuEBfLfvq0",
    "skills_vectorizer.pkl": "https://drive.google.com/uc?export=download&id=1ITP1paEQaTCla3SmV-RTliZCwjfXY8jG",
    "education_encoder.pkl": "https://drive.google.com/uc?export=download&id=1MqvOEGdeximjd-thffTcgrgcwRF40Yio",
    "interests_vectorizer.pkl": "https://drive.google.com/uc?export=download&id=1EaLpeIuwyp6uda6gE522lNYxH_KRb-wy",
}

def download_if_missing(filename, url):
    dest = os.path.join("Models", filename)
    if not os.path.exists(dest):
        st.info(f"Downloading {filename} ...")
        r = requests.get(url)
        with open(dest, "wb") as f:
            f.write(r.content)
    return dest

# Define dictionary with string keys
Models={}
Models["salary_model"] = joblib.load(download_if_missing("linear_regression_salary_model.joblib", MODEL_URLS["linear_regression_salary_model.joblib"]))
Models["career_recommendation_model"] = joblib.load(download_if_missing("career_recommendation_model.pkl", MODEL_URLS["career_recommendation_model.pkl"]))
Models["career_label_encoder"] = joblib.load(download_if_missing("career_label_encoder.pkl", MODEL_URLS["career_label_encoder.pkl"]))
Models["skills_vectorizer"] = joblib.load(download_if_missing("skills_vectorizer.pkl", MODEL_URLS["skills_vectorizer.pkl"]))
Models["education_encoder"] = joblib.load(download_if_missing("education_encoder.pkl", MODEL_URLS["education_encoder.pkl"]))
Models["interests_vectorizer"] = joblib.load(download_if_missing("interests_vectorizer.pkl", MODEL_URLS["interests_vectorizer.pkl"]))

# Optional: assign back to variables if your code expects them
salary_model = Models["salary_model"]
career_recommendation_model = Models["career_recommendation_model"]
career_label_encoder_obj = Models["career_label_encoder"]
skills_vectorizer = Models["skills_vectorizer"]
education_encoder = Models["education_encoder"]
interests_vectorizer = Models["interests_vectorizer"]



sys.modules["__main__"].semicolon_tokenizer = semicolon_tokenizer
sys.modules[__name__].semicolon_tokenizer = semicolon_tokenizer


EDUCATION_LEVELS = ["B.Tech", "High School", "M.Tech", "PhD"]
JOB_ROLES = ["Data Scientist", "Project Manager", "Software Engineer", "Data administrator"]
LOCATIONS = ["India", "UK", "USA","Remote"]
_EDU_CODE = {name: i for i, name in enumerate(EDUCATION_LEVELS)}
_ROLE_CODE = {name: i for i, name in enumerate(JOB_ROLES)}
_LOC_CODE = {name: i for i, name in enumerate(LOCATIONS)}


def build_career_features(Models, age, education, skills, interests):
    age_features = np.array([[age]])
    education_features = Models["education_encoder"].transform(
        pd.DataFrame({"Education": [education]})
    )
    skills_features = Models["skills_vectorizer"].transform([skills])
    interests_features = Models["interests_vectorizer"].transform([interests])
    return hstack(
        [age_features, education_features, skills_features, interests_features]
    )


import numpy as np
from scipy.sparse import csr_matrix, hstack

def predict_career(Models, age, education, skills_str, interests_str, debug=False):
    # Clean inputs
    skills_list = [s.strip() for s in skills_str.split(";") if s.strip()]
    interests_list = [i.strip() for i in interests_str.split(";") if i.strip()]

    # Vectorize text fields (vectorizers expect strings)
    skills_vec = Models["skills_vectorizer"].transform([";".join(skills_list)])  # sparse (1, V1)
    interests_vec = Models["interests_vectorizer"].transform([";".join(interests_list)])  # sparse (1, V2)

    # Encode education: could be scalar (LabelEncoder) or vector (OneHotEncoder)
    edu_encoded = Models["education_encoder"].transform([[education]])
    # edu_encoded might be numpy array or sparse matrix
    if hasattr(edu_encoded, "toarray"):
        edu_arr = edu_encoded.toarray()
    else:
        edu_arr = np.asarray(edu_encoded)

    # Decide how to treat education: scalar or vector
    if edu_arr.size == 1:
        edu_scalar = float(edu_arr.ravel()[0])
        edu_sparse = csr_matrix([[edu_scalar]])  # shape (1,1)
    else:
        # multi-column encoding: convert to sparse row
        edu_sparse = csr_matrix(edu_arr)  # shape (1, n_edu_cols)

    # Ensure age is a scalar number
    try:
        age_scalar = float(age)
    except Exception:
        # fallback: if age is array-like, extract first element
        age_scalar = float(np.asarray(age).ravel()[0])

    age_sparse = csr_matrix([[age_scalar]])  # shape (1,1)

    # Concatenate numerical + education + text vectors
    # Order must match training: [age, education_encoding..., skills_vec..., interests_vec...]
    X_new_sparse = hstack([age_sparse, edu_sparse, skills_vec, interests_vec])

    # Convert to dense if model requires dense input
    try:
        # many sklearn estimators accept sparse; if not, convert
        X_new = X_new_sparse.toarray()
    except Exception:
        X_new = X_new_sparse

    if debug:
        print("age_scalar:", age_scalar, type(age_scalar))
        print("edu_arr.shape:", edu_arr.shape)
        print("X_new shape:", getattr(X_new, "shape", None))
        if hasattr(Models["career_recommendation_model"], "n_features_in_"):
            print("model expects:", Models["career_recommendation_model"].n_features_in_)

    # Predict probabilities (use predict_proba if available)
    model = Models["career_recommendation_model"]
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_new)
        probabilities = probs[0]
    else:
        # fallback to decision_function or predict (normalize to pseudo-prob)
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X_new)
            # if binary, make into two-class softmax-like probabilities
            if scores.ndim == 1:
                scores = np.vstack([-scores, scores]).T
            exp = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            probabilities = (exp / exp.sum(axis=1, keepdims=True))[0]
        else:
            pred = model.predict(X_new)
            # map predicted class to probability 1.0
            classes = Models["career_label_encoder"].classes_
            probabilities = np.zeros(len(classes))
            idx = list(classes).index(pred[0])
            probabilities[idx] = 1.0

    # Rank careers
    classes = Models["career_label_encoder"].classes_
    ranked = sorted(zip(classes, probabilities), key=lambda x: -x[1])
    top_career = ranked[0][0]
    return top_career, ranked


def _code_for(value, code_map):
    """Look up the trained code for a value, or extend the mapping for a
    custom/'Other' value the model never saw during training. Custom
    values are an extrapolation and may be less reliable."""
    if value in code_map:
        return code_map[value]
    return max(code_map.values()) + 1


def predict_salary(Models, years_experience, education_level, job_role, location):
    X_new = pd.DataFrame(
        {
            "YearsExperience": [years_experience],
            "EducationLevel": [_EDU_CODE[education_level]],
            "JobRole": [_code_for(job_role, _ROLE_CODE)],
            "Location": [_code_for(location, _LOC_CODE)],
        }
    )
    return float(Models["salary_model"].predict(X_new)[0])


def format_inr(amount, assume_usd=True, usd_to_inr_rate=82.5):
    """
    Convert a numeric amount (default interpreted as USD) to INR and format with Indian grouping.
    - amount: int, float, or numeric string (e.g., 1500 or "1500")
    - assume_usd: if True, treat the input as USD and convert to INR; if False, treat input as already INR
    - usd_to_inr_rate: conversion rate to use when assume_usd is True
    Returns: formatted string like '₹1,23,750.00'
    """
    import re

    # Parse numeric input
    if isinstance(amount, str):
        s = amount.strip()
        # remove any non-numeric except minus and dot
        num_str = re.sub(r"[^\d\.\-]", "", s)
        if num_str == "" or num_str == "-" or num_str == ".":
            raise ValueError("amount must contain a numeric value")
        value = float(num_str)
    else:
        try:
            value = float(amount)
        except Exception:
            raise ValueError("amount must be numeric or numeric string")

    # Convert USD to INR if requested
    if assume_usd:
        value = value * float(usd_to_inr_rate)

    # Format with Indian digit grouping
    is_negative = value < 0
    whole, _, frac = f"{abs(value):.2f}".partition(".")
    if len(whole) > 3:
        last3, rest = whole[-3:], whole[:-3]
        parts = []
        while len(rest) > 2:
            parts.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.insert(0, rest)
        whole = ",".join(parts) + "," + last3
    return f"{'-' if is_negative else ''}₹{whole}.{frac}"


st.set_page_config(page_title="Career & Salary Predictor", layout="centered")
st.title("Career & Salary Predictor")


tab1, tab2 = st.tabs(["Career recommendation", "Salary prediction"])

with tab1:
    st.subheader("Tell us about yourself")
    age = st.number_input("Age", min_value=15, max_value=80, value=25, step=1)

    education_options = list(Models["education_encoder"].categories_[0])

    education = st.selectbox("Highest education level", education_options)

    skill_vocab = sorted(Models["skills_vectorizer"].vocabulary_.keys())
    interest_vocab = sorted(Models["interests_vectorizer"].vocabulary_.keys())

    skills_selected = st.multiselect("Skills", skill_vocab)
    interests_selected = st.multiselect("Interests", interest_vocab)

    if st.button("Recommend career", type="primary"):
        if not skills_selected or not interests_selected:
            st.warning("Pick at least one skill and one interest.")
        else:
            skills_str = ";".join(skills_selected)
            interests_str = ";".join(interests_selected)
            top_career, ranked = predict_career(
                Models, age, education, skills_str, interests_str
            )

            st.success(f"Recommended career: **{top_career}**")
            st.write("Top matches:")
            for career, prob in ranked:
                st.write(f"{career} — {prob * 100:.1f}%")
                st.progress(min(max(prob, 0.0), 1.0))

with tab2:
    st.subheader("Salary details")
    years = st.number_input(
        "Years of experience", min_value=0.0, max_value=50.0, value=2.0, step=0.5
    )
    education_level = st.selectbox("Education level", EDUCATION_LEVELS)

    job_role_choice = st.selectbox("Job role", JOB_ROLES + ["Other"])
    if job_role_choice == "Other":
        job_role = st.text_input("Type the job role")
    else:
        job_role = job_role_choice

    location_choice = st.selectbox("Location", LOCATIONS + ["Other"])
    if location_choice == "Other":
        location = st.text_input("Type the location")
    else:
        location = location_choice

    if job_role_choice == "Other" or location_choice == "Other":
        st.caption(
            "'Other' entries weren't in the training data, so the prediction "
            "for them is an extrapolation and may be less reliable."
        )

    period = st.radio(
        "The model was trained to predict salary as:",
        ["Annual", "Monthly"],
        horizontal=True,
    )

    if st.button("Predict salary", type="primary"):
        if job_role_choice == "Other" and not job_role.strip():
            st.warning("Type a job role.")
        elif location_choice == "Other" and not location.strip():
            st.warning("Type a location.")
        else:
            salary = predict_salary(Models, years, education_level, job_role, location)
            if period == "Annual":
                annual, monthly = salary, salary / 12
            else:
                monthly, annual = salary, salary * 12
            lpa = annual / 100000
 
            st.success(
                f"Predicted salary: **{format_inr(annual)} / year "
                f"({lpa:.2f} LPA)**\n\n"
                f"≈ {format_inr(monthly)} / month"
            )
