import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Auto-detect model path
if os.path.exists("models/rf_pipeline.pkl"):
    model_path = "models/rf_pipeline.pkl"
elif os.path.exists("rf_pipeline.pkl"):
    model_path = "rf_pipeline.pkl"
else:
    st.error("❌ Model file not found! Place rf_pipeline.pkl inside a folder named 'models'.")
    st.stop()

try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.stop()

st.title("Heart Disease Prediction")
st.write("Provide patient information to predict presence of heart disease (binary).")

numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
object_cols = ['sex', 'dataset', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

user_input = {}

for col in numeric_cols:
    user_input[col] = st.number_input(col, value=0.0)
input_df = pd.DataFrame([user_input])

st.write("Input:")
st.write(input_df)

try:
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0, 1]
    
    st.write("Prediction (0 = No disease, 1 = Disease):", int(pred))
    st.write("Probability of disease:", float(proba))
except Exception as e:
    st.error(f"❌ Error making prediction: {str(e)}")

