import streamlit as st
import numpy as np
import joblib
import pefile
import tempfile

# Load model
try:
    with open("rf_model.pkl", "rb") as file:
        model = joblib.load(file)
except FileNotFoundError:
    st.error("Model file not found. Upload rf_model.pkl.")
    st.stop()

st.title("🛡️ Ransomware Detection System")
st.markdown("Upload a PE file (.exe or .dll) to detect if it's ransomware or benign.")

uploaded_file = st.file_uploader("Upload PE File", type=none)

def extract_features(pe):
    return [
        pe.OPTIONAL_HEADER.DATA_DIRECTORY[6].VirtualAddress,  # DebugRVA
        pe.FILE_HEADER.Machine,
        pe.OPTIONAL_HEADER.MajorOperatingSystemVersion,
        pe.OPTIONAL_HEADER.MajorLinkerVersion,
        pe.OPTIONAL_HEADER.DllCharacteristics,
        pe.OPTIONAL_HEADER.DATA_DIRECTORY[12].VirtualAddress,  # IatVRA
        pe.OPTIONAL_HEADER.MajorImageVersion
    ]

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    try:
        pe = pefile.PE(temp_file_path)
        features = np.array([extract_features(pe)])
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0]

        st.subheader("🔍 File Analysis Result")

        if prediction == 0:
            st.error("🚨 Detected: RANSOMWARE")
            st.markdown(f"**Confidence:** {proba[0]*100:.2f}% ransomware")
        else:
            st.success("✅ Detected: BENIGN")
            st.markdown(f"**Confidence:** {proba[1]*100:.2f}% benign")

        with st.expander("🔧 Extracted Features"):
            feature_names = [
                "DebugRVA", "Machine", "MajorOSVersion",
                "MajorLinkerVersion", "DllCharacteristics",
                "IatVRA", "MajorImageVersion"
            ]
            for name, value in zip(feature_names, features[0]):
                st.write(f"**{name}**: {value}")

    except Exception as e:
        st.error(f"Error parsing PE file: {e}")
