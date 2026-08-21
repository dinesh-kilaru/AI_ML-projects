
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


def predict_career(Models, age, education, skills, interests, top_n=5):
    X_new = build_career_features(Models, age, education, skills, interests)
    prediction = Models["career_model"].predict(X_new)
    top_career = Models["career_label_encoder"].inverse_transform(prediction)[0]
   prediction = Models["career_recommendation_model"].predict(X_new)[0]

    ranked = sorted(
        zip(Models["career_label_encoder"].classes_, probabilities),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return top_career, ranked[:top_n]


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
                st.success(
                    f"Predicted salary: **${salary:,.2f} / year** "
                    f"(≈ ${salary / 12:,.2f} / month)"
                )
            else:
                st.success(
                    f"Predicted salary: **${salary:,.2f} / month** "
                    f"(≈ ${salary * 12:,.2f} / year)"
                )
