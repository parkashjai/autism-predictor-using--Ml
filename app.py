import streamlit as st
import pandas as pd
import pickle
import time

# Load model
model = pickle.load(open("best_model.pkl", "rb"))
encoders = pickle.load(open("encoder.pkl", "rb"))

st.set_page_config(page_title="ASD Predictor")

st.title("🧠 ASD Prediction App")

# Sidebar inputs
A1 = st.sidebar.selectbox("A1", ["Yes", "No"])
A2 = st.sidebar.selectbox("A2", ["Yes", "No"])
A3 = st.sidebar.selectbox("A3", ["Yes", "No"])
A4 = st.sidebar.selectbox("A4", ["Yes", "No"])
A5 = st.sidebar.selectbox("A5", ["Yes", "No"])
A6 = st.sidebar.selectbox("A6", ["Yes", "No"])
A7 = st.sidebar.selectbox("A7", ["Yes", "No"])
A8 = st.sidebar.selectbox("A8", ["Yes", "No"])
A9 = st.sidebar.selectbox("A9", ["Yes", "No"])
A10 = st.sidebar.selectbox("A10", ["Yes", "No"])

age = st.sidebar.slider("Age", 1, 100, 25)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

# Safe dropdowns
ethnicity = st.sidebar.selectbox(
    "Ethnicity",
    encoders["ethnicity"].classes_ if "ethnicity" in encoders else ["Unknown"]
)

country = st.sidebar.selectbox(
    "Country",
    encoders["country_of_res"].classes_ if "country_of_res" in encoders else ["Unknown"]
)

jaundice = st.sidebar.selectbox("Jaundice", ["Yes", "No"])
autism = st.sidebar.selectbox("Autism Family", ["Yes", "No"])
used_app = st.sidebar.selectbox("Used App Before", ["Yes", "No"])
result = st.sidebar.slider("Score", 0, 10, 5)

# Dataframe
data = pd.DataFrame({
    "A1_Score":[A1], "A2_Score":[A2], "A3_Score":[A3], "A4_Score":[A4],
    "A5_Score":[A5], "A6_Score":[A6], "A7_Score":[A7], "A8_Score":[A8],
    "A9_Score":[A9], "A10_Score":[A10],
    "age":[age],
    "gender":[gender],
    "ethnicity":[ethnicity],
    "jaundice":[jaundice],
    "autism":[autism],
    "country_of_res":[country],
    "used_app_before":[used_app],
    "result":[result]
})

# 🔥 Encoding
for col in encoders:
    if col in data.columns:
        try:
            data[col] = encoders[col].transform(data[col])
        except:
            data[col] = 0

# 🔥 Yes/No mapping
yes_no_map = {"Yes": 1, "No": 0}
for col in data.columns:
    if data[col].dtype == "object":
        data[col] = data[col].map(yes_no_map)

# 🔥 Convert to numeric (FINAL FIX)
data = data.apply(pd.to_numeric, errors='coerce').fillna(0)

# 🔥 Fix column order
data = data.reindex(columns=model.feature_names_in_, fill_value=0)

# Button
if st.button("Predict"):
    bar = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        bar.progress(i+1)

    pred = model.predict(data)[0]
    proba = model.predict_proba(data)[0]

    if pred == 1:
        st.error("⚠️ Autism Detected")
    else:
        st.success("✅ No Autism")

    st.write(f"ASD Probability: {proba[1]*100:.2f}%")
    st.progress(float(proba[1]))