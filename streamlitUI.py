
import streamlit as st
import numpy as np
import joblib


model = joblib.load("rf_model.pkl")  # or however you're loading it

st.title("🛡️ Ransomware Detection System")
st.markdown("Enter extracted PE file features to classify as benign or ransomware.")

# Feature inputs — adjust based on your model's features
debug_rva = st.number_input("DebugRVA", value=0)
machine = st.number_input("Machine", value=332)
major_os = st.number_input("MajorOSVersion", value=6)
linker_version = st.number_input("MajorLinkerVersion", value=9)
dll_char = st.number_input("DllCharacteristics", value=3440)
iat_vra = st.number_input("IatVRA", value=1024)
image_version = st.number_input("MajorImageVersion", value=1)

if st.button("Predict"):
    features = np.array([[debug_rva, machine, major_os, linker_version, dll_char, iat_vra, image_version]])
    prediction = model.predict(features)[0]

    if prediction == 1:
        st.error("🚨 Detected: RANSOMWARE")
    else:
        st.success("✅ Detected: BENIGN")
