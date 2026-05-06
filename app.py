import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from textblob import TextBlob
import plotly.graph_objects as go
import os

import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --- Page Configuration ---
st.set_page_config(page_title="Aman Credit Scoring", layout="wide")

# --- Function to Load Model and Scalers ---
@st.cache_resource
def load_artifacts():
    # Make sure these filenames match your folder exactly
    model_path = 'credit_model.h5'
    scaler_path = 'scaler.joblib' 
    features_path = 'feature_names.joblib'
    
    # Check if files exist to avoid crash
    if not os.path.exists(model_path):
        st.error(f"❌ Error: {model_path} not found in the directory.")
        return None, None, None
    
    # Load the trained model and processing tools
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(features_path)
    return model, scaler, feature_names

# --- Load Data ---
model, scaler, feature_names = load_artifacts()

if model:
    st.title("💳 Aman Credit Scoring Dashboard")
    st.markdown("---")
    
    # --- Sidebar Input Section ---
    st.sidebar.header("📊 Customer Input Data")
    
    user_inputs = {}
    # Dynamically create inputs based on the feature names list
    for col in feature_names:
        if col == 'sentiment_polarity':
            user_inputs[col] = st.sidebar.slider("Sentiment Analysis Score", -1.0, 1.0, 0.0)
        else:
            user_inputs[col] = st.sidebar.number_input(f"{col}", value=0.0)

    # --- Prediction Logic ---
    if st.sidebar.button("🚀 Calculate Credit Score"):
        # Convert inputs to DataFrame with correct feature order
        input_df = pd.DataFrame([user_inputs])[feature_names]
        
        # 1. Scale the data
        input_scaled = scaler.transform(input_df)
        
        # 2. Get prediction from TensorFlow model
        prediction = float(model.predict(input_scaled)[0][0])
        
        # --- Display Results ---
        st.balloons()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric(label="Final Credit Score", value=f"{prediction:.2f}/100")
            if prediction >= 70:
                st.success("Status: Safe / Low Risk")
            elif prediction >= 40:
                st.warning("Status: Moderate Risk")
            else:
                st.error("Status: High Risk / Risky")
        
        with col2:
            # Gauge Chart for visual representation
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prediction,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Credit Risk Level"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#003366"},
                    'steps': [
                        {'range': [0, 40], 'color': "#ffcccc"},
                        {'range': [40, 70], 'color': "#fff0cc"},
                        {'range': [70, 100], 'color': "#cce5cc"}
                    ]
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

else:
    # Error message if files are missing
    st.warning("System Halted: Please ensure 'credit_model.h5', 'scaler.joblib', and 'feature_names.joblib' are in the same folder.")

# --- Footer ---
st.markdown("---")
st.caption("© 2026 Aman Financial Services | AI Powered Credit Intelligence")
