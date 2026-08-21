
import sys
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from scipy.sparse import hstack

BASE_DIR_DEFAULT = "C:/Users/dines/OneDrive/Desktop/ry prediction"

FILES = {
    "career_model": "career_recommendation_model.pkl",
    "career_label_encoder": "career_label_encoder.pkl",
    "skills_vectorizer": "skills_vectorizer.pkl",
    "interests_vectorizer": "interests_vectorizer.pkl",
    "education_encoder": "education_encoder.pkl",
    "salary_model": "linear_regression_salary_model.joblib",
}


def semicolon_tokenizer(text):
    """'python;sql;machine learning' -> ['python', 'sql', 'machine learning']"""
    return text.split(";")

sys.modules["__main__"].semicolon_tokenizer = semicolon_tokenizer
sys.modules[__name__].semicolon_tokenizer = semicolon_tokenizer


EDUCATION_LEVELS = ["Bachelors", "High School", "Masters", "PhD"]
JOB_ROLES = ["Data Scientist", "Project Manager", "Software Engineer", "Data administrator"]
LOCATIONS = ["India", "UK", "USA","Remote"]
_EDU_CODE = {name: i for i, name in enumerate(EDUCATION_LEVELS)}
_ROLE_CODE = {name: i for i, name in enumerate(JOB_ROLES)}
_LOC_CODE = {name: i for i, name in enumerate(LOCATIONS)}


@st.cache_resource(show_spinner="Loading models...")
def load_models(base_dir):
    models = {}
    for key, filename in FILES.items():
        path = f"{base_dir}/{filename}"
        models[key] = joblib.load(path)
    return models


def build_career_features(models, age, education, skills, interests):
    age_features = np.array([[age]])
    education_features = models["education_encoder"].transform(
        pd.DataFrame({"Education": [education]})
    )
    skills_features = models["skills_vectorizer"].transform([skills])
    interests_features = models["interests_vectorizer"].transform([interests])
    return hstack(
        [age_features, education_features, skills_features, interests_features]
    )


def predict_career(models, age, education, skills, interests, top_n=5):
    X_new = build_career_features(models, age, education, skills, interests)
    prediction = models["career_model"].predict(X_new)
    top_career = models["career_label_encoder"].inverse_transform(prediction)[0]
    probabilities = models["career_model"].predict_proba(X_new)[0]
    ranked = sorted(
        zip(models["career_label_encoder"].classes_, probabilities),
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


def predict_salary(models, years_experience, education_level, job_role, location):
    X_new = pd.DataFrame(
        {
            "YearsExperience": [years_experience],
            "EducationLevel": [_EDU_CODE[education_level]],
            "JobRole": [_code_for(job_role, _ROLE_CODE)],
            "Location": [_code_for(location, _LOC_CODE)],
        }
    )
    return float(models["salary_model"].predict(X_new)[0])


st.set_page_config(page_title="Career & Salary Predictor", layout="centered")
st.title("Career & Salary Predictor")

try:
    models = load_models(BASE_DIR_DEFAULT)
except FileNotFoundError as e:
    st.error(f"Couldn't find one of the model files in that folder.\n\n{e}")
    st.stop()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

tab1, tab2 = st.tabs(["Career recommendation", "Salary prediction"])

with tab1:
    st.subheader("Tell us about yourself")
    age = st.number_input("Age", min_value=15, max_value=80, value=25, step=1)

    education_options = list(models["education_encoder"].categories_[0])
    education = st.selectbox("Highest education level", education_options)

    skill_vocab = sorted(models["skills_vectorizer"].vocabulary_.keys())
    interest_vocab = sorted(models["interests_vectorizer"].vocabulary_.keys())

    skills_selected = st.multiselect("Skills", skill_vocab)
    interests_selected = st.multiselect("Interests", interest_vocab)

    if st.button("Recommend career", type="primary"):
        if not skills_selected or not interests_selected:
            st.warning("Pick at least one skill and one interest.")
        else:
            skills_str = ";".join(skills_selected)
            interests_str = ";".join(interests_selected)
            top_career, ranked = predict_career(
                models, age, education, skills_str, interests_str
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
            salary = predict_salary(models, years, education_level, job_role, location)
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