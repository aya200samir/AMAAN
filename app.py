# app.py - Aman Credit Scoring Dashboard
# Professional UI/UX with HTML/CSS injection, Plotly gauge, SHAP, and comparison table

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from textblob import TextBlob
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap
from PIL import Image
import base64

# ------------------------------
# 1. Configure page (wide layout, title, icon)
# ------------------------------
st.set_page_config(page_title="Aman Credit Scoring", page_icon="💳", layout="wide")

# ------------------------------
# 2. Custom CSS (brand colors, fonts, sidebars, cards)
# ------------------------------
st.markdown("""
<style>
    /* Main brand colors */
    :root {
        --navy: #003366;
        --orange: #FF9900;
        --light-bg: #F8F9FA;
        --card-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    /* Sidebar styling */
    .css-1d391kg, .css-12oz5g0 {
        background-color: var(--navy);
    }
    .sidebar .sidebar-content {
        background-color: var(--navy);
        color: white;
    }
    /* Override Streamlit's default text in sidebar */
    .sidebar .sidebar-content .stMarkdown, .sidebar .sidebar-content label {
        color: white !important;
    }
    /* Buttons */
    .stButton button {
        background-color: var(--orange);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton button:hover {
        background-color: #cc7a00;
        color: white;
        transform: scale(1.02);
    }
    /* Headers and text */
    h1, h2, h3, h4 {
        font-family: 'Cairo', 'Segoe UI', sans-serif;
        color: var(--navy);
    }
    /* Cards for results */
    .card {
        background-color: white;
        border-radius: 16px;
        box-shadow: var(--card-shadow);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    /* Decision summary box */
    .decision-box {
        background-color: #f0f2f6;
        border-left: 6px solid var(--orange);
        border-radius: 12px;
        padding: 1rem;
        font-size: 1rem;
    }
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #e0e0e0;
        font-size: 0.8rem;
        color: #6c757d;
    }
    /* Tooltip style (using title attribute) */
    .tooltip-icon {
        border-bottom: 1px dashed var(--orange);
        cursor: help;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------
# 3. Load artifacts (cached for performance)
# ------------------------------
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model('credit_model.h5')
    scaler = joblib.load('scaler.joblib')
    feature_names = joblib.load('feature_names.joblib')
    # Load safe average (if exists, else compute a dummy)
    try:
        safe_avg = joblib.load('average_safe_customer.joblib')
    except:
        safe_avg = {}  # will be handled later
    return model, scaler, feature_names, safe_avg

model, scaler, feature_names, safe_avg = load_artifacts()

# If safe_avg is empty, compute a default baseline (use zeros)
if not safe_avg:
    safe_avg = {f: 0.0 for f in feature_names}

# ------------------------------
# 4. Helper functions (sentiment, penalty, gauge, shap explanation)
# ------------------------------
def get_sentiment_polarity(text):
    if not text or text.strip() == "":
        return 0.0
    try:
        return TextBlob(text).sentiment.polarity
    except:
        return 0.0

def apply_risk_penalty(score, sentiment, threshold=-0.5, penalty=5):
    return max(0, min(100, score - penalty)) if sentiment < threshold else score

def plot_gauge(score):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        title = {'text': "Credit Score"},
        domain = {'x': [0,1], 'y': [0,1]},
        gauge = {
            'axis': {'range': [0,100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#003366"},
            'steps': [
                {'range': [0,40], 'color': '#ffcccc'},
                {'range': [40,70], 'color': '#fff0cc'},
                {'range': [70,100], 'color': '#cce5cc'}
            ],
            'threshold': {
                'line': {'color': "#FF9900", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20,r=20,t=40,b=20))
    return fig

def get_shap_bar_plot(model, input_scaled, feature_names, background_data=None):
    """Returns matplotlib figure of SHAP bar plot"""
    # Use KernelExplainer (works with any model)
    if background_data is None:
        # Generate dummy background for demo (in production use real training sample)
        background_data = np.random.randn(50, len(feature_names))
    def predict_fn(x):
        return model.predict(x).flatten()
    explainer = shap.KernelExplainer(predict_fn, background_data)
    shap_values = explainer.shap_values(input_scaled)
    plt.figure(figsize=(8, 4))
    shap.summary_plot(shap_values, input_scaled, feature_names=feature_names, plot_type="bar", show=False)
    fig = plt.gcf()
    plt.close()
    return fig

# ------------------------------
# 5. Sidebar: inputs grouped with tooltips & brand color
# ------------------------------
st.sidebar.markdown(f"""
<div style="text-align:center; padding:1rem 0;">
    <img src="data:image/png;base64,{base64.b64encode(open('logo.png', 'rb').read()).decode()}" width="150" style="filter: brightness(0) invert(1);">
</div>
""" if os.path.exists('logo.png') else "<h2 style='color:white;'>✨ AMAN</h2>", unsafe_allow_html=True)

st.sidebar.markdown("<h3 style='color:white; margin-top:0;'>📊 Customer Profile</h3>", unsafe_allow_html=True)

with st.sidebar.expander("💰 Financial Indicators", expanded=True):
    amount = st.number_input("Transaction Amount (USD)", min_value=0.0, max_value=10000.0, value=500.0,
                             help="Amount of the latest transaction")
    oldbalanceOrg = st.number_input("Old Balance (Original Account)", min_value=0.0, max_value=50000.0, value=2000.0,
                                    help="Balance before the transaction")
    # Add other financial features if needed (like newbalance, etc.)
    # For simplicity we keep core ones; you can extend

with st.sidebar.expander("🛒 Purchase Behavior", expanded=True):
    recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=365, value=30,
                              help="Days since last order")
    frequency = st.number_input("Frequency (number of orders)", min_value=1, max_value=50, value=5,
                                help="Total orders in history")
    monetary = st.number_input("Monetary (total spend)", min_value=0.0, max_value=5000.0, value=500.0,
                               help="Total amount spent")

# Additional features placeholder (if your model has more)
other_features = {}
remaining_features = [f for f in feature_names if f not in ['amount', 'oldbalanceOrg', 'Recency_days', 'Frequency', 'Monetary', 'sentiment_polarity']]
if remaining_features:
    with st.sidebar.expander("🔧 Other Features", expanded=False):
        for fname in remaining_features:
            other_features[fname] = st.number_input(f"{fname.replace('_',' ').title()}", value=0.0, step=0.01)

# Review text area
review_text = st.sidebar.text_area("💬 Customer Review (for sentiment analysis)", height=120,
                                   placeholder="e.g., 'Excellent service, very fast delivery!'")

predict_btn = st.sidebar.button("🚀 Predict Credit Score", use_container_width=True)

# ------------------------------
# 6. Main area: results after prediction
# ------------------------------
if predict_btn:
    with st.spinner("Calculating credit score and explanations..."):
        # a. Sentiment
        sentiment = get_sentiment_polarity(review_text)

        # b. Build input dictionary
        input_dict = {
            'amount': amount,
            'oldbalanceOrg': oldbalanceOrg,
            'Recency_days': recency,
            'Frequency': frequency,
            'Monetary': monetary,
            'sentiment_polarity': sentiment
        }
        input_dict.update(other_features)
        # Ensure all features are present
        for f in feature_names:
            if f not in input_dict:
                input_dict[f] = 0.0  # default fallback

        input_df = pd.DataFrame([input_dict])[feature_names]

        # c. Scale
        input_scaled = scaler.transform(input_df)

        # d. Raw prediction
        raw_pred = float(model.predict(input_scaled)[0][0])

        # e. Apply risk penalty
        final_score = apply_risk_penalty(raw_pred, sentiment)

        # f. Category
        if final_score >= 70:
            category_html = "<span style='color:#28a745;font-weight:bold'>✅ Safe</span>"
        elif final_score >= 40:
            category_html = "<span style='color:#ffc107;font-weight:bold'>⚠️ Moderate</span>"
        else:
            category_html = "<span style='color:#dc3545;font-weight:bold'>🔴 Risky</span>"

    # ------------------------------
    # Display results in columns
    # ------------------------------
    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.plotly_chart(plot_gauge(final_score), use_container_width=True)
        st.metric("Raw Model Score", f"{raw_pred:.1f}")
        st.metric("Sentiment Polarity", f"{sentiment:.3f}")
        if sentiment < -0.5:
            st.warning(f"⚠️ Risk penalty applied: -5 points (sentiment {sentiment:.2f} < -0.5)")
        else:
            st.success("✅ No risk penalty.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🔍 Model Explanation")
        # SHAP bar plot (using a dummy background; in production replace with real training sample)
        background_sample = np.random.randn(50, len(feature_names))  # placeholder
        shap_fig = get_shap_bar_plot(model, input_scaled, feature_names, background_sample)
        st.pyplot(shap_fig)
        st.markdown("</div>", unsafe_allow_html=True)

    # Comparison table with safe customer average
    st.markdown("<h3>📋 Comparison vs. Safe Customer</h3>", unsafe_allow_html=True)
    safe_series = pd.Series(safe_avg)
    customer_series = pd.Series(input_dict)
    compare_df = pd.DataFrame({
        'Feature': safe_series.index,
        'Safe Customer Average': safe_series.values,
        'Current Customer': customer_series.values,
        'Difference': customer_series.values - safe_series.values
    })
    st.dataframe(compare_df.style.format({
        'Safe Customer Average': '{:.2f}',
        'Current Customer': '{:.2f}',
        'Difference': '{:.2f}'
    }).background_gradient(cmap='RdYlGn', subset=['Difference']), use_container_width=True)

    # Decision summary box (HTML/CSS)
    st.markdown(f"""
    <div class='decision-box'>
        <strong>📢 Credit Decision Summary</strong><br>
        <strong>Final Score:</strong> {final_score:.0f} {category_html}<br>
        <strong>Sentiment impact:</strong> polarity = {sentiment:.2f} → {"Risk penalty applied (-5)" if sentiment < -0.5 else "No penalty"}<br>
        <strong>Top driving factors:</strong> (see SHAP plot).<br>
        <em>This decision is based on AI model and is for informational purposes.</em>
    </div>
    """, unsafe_allow_html=True)

else:
    st.info("👈 Please fill in the customer details in the sidebar and click 'Predict Credit Score'.")

# ------------------------------
# 7. Footer (HTML)
# ------------------------------
st.markdown("""
<hr>
<div class='footer'>
    © 2026 Aman Financial Services | AI Credit Intelligence | Powered by Deep Learning & SHAP
</div>
""", unsafe_allow_html=True)
