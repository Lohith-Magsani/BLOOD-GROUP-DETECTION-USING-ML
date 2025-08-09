import streamlit as st
import tempfile
import os
from predict import predict_blood_group

st.title("Blood Group Prediction from Fingerprint")
uploaded_file = st.file_uploader("Upload fingerprint image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        # Create a temporary file to store the uploaded image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_file_path = temp_file.name

        # Predict the blood group
        prediction = predict_blood_group(temp_file_path)

        # Display the result
        st.success(f"Predicted Blood Group: {prediction}")

        # Clean up the temporary file
        os.remove(temp_file_path)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
