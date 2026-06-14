import streamlit as st
import pickle
import pandas as pd
import random
import os
import time

# -----------------------------
# Page Title
# -----------------------------
st.header("Heart Disease Prediction Using Machine Learning")

# -----------------------------
# Project Description
# -----------------------------
data = """
Project Objective

Heart Disease Prediction using Machine Learning

Heart disease prevention is critical, and data-driven prediction systems can significantly aid in early diagnosis and treatment. Machine Learning offers accurate prediction capabilities, enhancing healthcare outcomes.

In this project, a heart disease dataset was analyzed with appropriate preprocessing. Multiple classification algorithms were implemented in Python using Scikit-learn and Keras to predict the presence of heart disease.

Algorithms Used:
- Logistic Regression
- Naive Bayes
- Support Vector Machine (Linear)
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- XGBoost
- Artificial Neural Network (1 Hidden Layer, Keras)
"""

# -----------------------------
# Load Model
# -----------------------------
MODEL_PATH = os.path.join("ml_model", "Heart_disease_pred.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# -----------------------------
# Load Dataset
# -----------------------------
try:
    df = pd.read_csv("heart.csv")
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# -----------------------------
# Display Description
# -----------------------------
st.subheader(data)

# -----------------------------
# Images
# -----------------------------
st.image("https://t-shikuro.github.io/images/heart/heart.gif")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Select Features to Predict Heart Disease")

st.sidebar.image(
    "https://tse4.mm.bing.net/th/id/OIP.7LA1z7w-drtQmnFmC0KBNAHaE7?cb=thfvnext&pid=ImgDet&w=201&h=134&c=7&o=7&rm=3"
)

# -----------------------------
# User Inputs
# -----------------------------
all_values = []

feature_columns = df.columns[:-1]

for col in feature_columns:
    min_value = int(df[col].min())
    max_value = int(df[col].max())

    value = st.sidebar.slider(
        f"Select {col}",
        min_value,
        max_value,
        random.randint(min_value, max_value)
    )

    all_values.append(value)

# -----------------------------
# Prediction Button
# -----------------------------
if st.sidebar.button("Predict"):

    final_value = [all_values]

    try:
        prediction = model.predict(final_value)[0]
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.stop()

    progress_bar = st.progress(0)

    placeholder = st.empty()
    placeholder.subheader("Predicting Heart Disease...")

    place = st.empty()
    place.image(
        "https://media1.tenor.com/m/LLlSFiqwJGMAAAAC/beating-heart-gif.gif",
        width=200
    )

    for i in range(100):
        time.sleep(0.02)
        progress_bar.progress(i + 1)

    placeholder.empty()
    place.empty()

    if prediction == 0:
        st.success("✅ No Heart Disease Detected")
    else:
        st.warning("⚠️ Heart Disease Detected")