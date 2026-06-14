import os
import pickle
import pandas as pd
import streamlit as st
import random
import time

st.header("Heart Disease Prediction Using Machine Learning")

data = '''Project Objective
Heart Disease Prediction using Machine Learning
Heart disease prevention is critical, and data-driven prediction systems can significantly aid in early diagnosis and treatment. Machine Learning offers accurate prediction capabilities, enhancing healthcare outcomes.
In this project, I analyzed a heart disease dataset with appropriate preprocessing. Multiple classification algorithms were implemented in Python using Scikit-learn and Keras to predict the presence of heart disease.

Algorithms Used:
- Logistic Regression
- Naive Bayes
- Support Vector Machine (Linear)
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- XGBoost
- Artificial Neural Network (1 Hidden Layer, Keras)'''

# Path to the .pkl file inside ml_model
model_path = os.path.join(os.path.dirname(__file__), "ml_model", "Heart_disease_pred.pkl")

# Load the model
with open(model_path, "rb") as f:
    chatgpt = pickle.load(f)

st.subheader(data)

st.image('https://t-shikuro.github.io/images/heart/heart.gif')

# ✅ Load CSV from the same folder
csv_path = os.path.join(os.path.dirname(__file__), "heart.csv")
df = pd.read_csv(csv_path)

st.sidebar.header("Select feature to predict heart disease")
st.sidebar.image("https://tse4.mm.bing.net/th/id/OIP.7LA1z7w-drtQmnFmC0KBNAHaE7?cb=thfvnext&pid=ImgDet&w=201&h=134&c=7&o=7&rm=3")

all_values = []
random.seed(11)
for i in df.iloc[:, :-1]:
    min_value, max_value = df[i].agg(['min', 'max'])
    var = st.sidebar.slider(f'Select {i} value', int(min_value), int(max_value),
                            random.randint(int(min_value), int(max_value)))
    all_values.append(var)

final_value = [all_values]
ans = chatgpt.predict(final_value)[0]

progress_bar = st.progress(0)
placeholder = st.empty()
placeholder.subheader('Predicting Heart Disease')

place = st.empty()
place.image('https://media1.tenor.com/m/LLlSFiqwJGMAAAAC/beating-heart-gif.gif', width=200)

for i in range(100):
    time.sleep(0.05)
    progress_bar.progress(i + 1)

if ans == 0:
    body = f'No Heart Disease Detected'
    placeholder.empty()
    place.empty()
    st.success(body)
else:
    body = 'Heart Disease Found'
    placeholder.empty()
    place.empty()
    st.warning(body)

st.markdown("Design By : Divynashu Raj")
