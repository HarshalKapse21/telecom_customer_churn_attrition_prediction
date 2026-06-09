import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📡",
    layout="wide",
)

# ── Load artifacts ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = pickle.load(open("models/random_forest_model.pkl", "rb"))
scaler = pickle.load(open("models/standard_scaler.pkl", "rb"))

    with open("models/meta.json", "r") as f:
        meta = json.load(f)

    return model, scaler, meta

# ── Header ──────────────────────────────────────────────────────────────────────
st.title("📡 Telco Customer Churn Predictor")
st.markdown(
    "Fill in the customer details below and click **Predict** to see whether "
    "this customer is likely to churn."
)
st.divider()

# ── Input form ──────────────────────────────────────────────────────────────────
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Customer Info")
        senior_citizen   = st.selectbox("Senior Citizen",   ["No", "Yes"])
        partner          = st.selectbox("Has Partner",       ["No", "Yes"])
        dependents       = st.selectbox("Has Dependents",    ["No", "Yes"])
        tenure           = st.slider("Tenure (months)",      0, 72, 12)

    with col2:
        st.subheader("🔒 Services")
        online_security  = st.selectbox("Online Security",   ["No", "No internet service", "Yes"])
        online_backup    = st.selectbox("Online Backup",     ["No", "No internet service", "Yes"])
        device_protect   = st.selectbox("Device Protection", ["No", "No internet service", "Yes"])
        tech_support     = st.selectbox("Tech Support",      ["No", "No internet service", "Yes"])

    with col3:
        st.subheader("💳 Billing")
        contract         = st.selectbox("Contract Type",     ["Month-to-month", "One year", "Two year"])
        paperless        = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method   = st.selectbox("Payment Method",    [
            "Bank transfer (automatic)", "Credit card (automatic)",
            "Electronic check", "Mailed check"
        ])
        monthly_charges  = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, step=0.5)
        total_charges    = st.number_input("Total Charges ($)",    0.0, 10000.0, monthly_charges * tenure, step=1.0)

    submitted = st.form_submit_button("🔍 Predict Churn", use_container_width=True)

# ── Prediction ──────────────────────────────────────────────────────────────────
def encode(feature, value):
    return le_mappings[feature][value]

if submitted:
    # Build raw input dict (only final_features used by the model)
    raw = {
        "SeniorCitizen":    1 if senior_citizen == "Yes" else 0,
        "Partner":          encode("Partner",          partner),
        "Dependents":       encode("Dependents",       dependents),
        "tenure":           tenure,
        "OnlineSecurity":   encode("OnlineSecurity",   online_security),
        "OnlineBackup":     encode("OnlineBackup",     online_backup),
        "DeviceProtection": encode("DeviceProtection", device_protect),
        "TechSupport":      encode("TechSupport",      tech_support),
        "Contract":         encode("Contract",         contract),
        "PaperlessBilling": encode("PaperlessBilling", paperless),
        "PaymentMethod":    encode("PaymentMethod",    payment_method),
        "MonthlyCharges":   monthly_charges,
        "TotalCharges":     total_charges,
    }

    input_df = pd.DataFrame([raw])[final_features]
    input_scaled = scaler.transform(input_df)

    prediction  = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    churn_prob  = probability[1] * 100
    stay_prob   = probability[0] * 100

    st.divider()
    st.subheader("🎯 Prediction Result")

    res_col1, res_col2, res_col3 = st.columns(3)

    with res_col1:
        if prediction == 1:
            st.error("⚠️ **HIGH CHURN RISK**\nThis customer is likely to churn.")
        else:
            st.success("✅ **LOW CHURN RISK**\nThis customer is likely to stay.")

    with res_col2:
        st.metric("Churn Probability",  f"{churn_prob:.1f}%")
        st.metric("Retention Probability", f"{stay_prob:.1f}%")

    with res_col3:
        # Risk breakdown
        if churn_prob >= 70:
            risk_label, risk_color = "🔴 High Risk", "red"
        elif churn_prob >= 40:
            risk_label, risk_color = "🟡 Medium Risk", "orange"
        else:
            risk_label, risk_color = "🟢 Low Risk", "green"
        st.markdown(f"### Risk Level\n# {risk_label}")

    # Key factors
    st.divider()
    st.subheader("📋 Key Customer Snapshot")
    snap_col1, snap_col2, snap_col3, snap_col4 = st.columns(4)
    snap_col1.metric("Tenure",           f"{tenure} months")
    snap_col2.metric("Monthly Charges",  f"${monthly_charges:.2f}")
    snap_col3.metric("Contract Type",    contract)
    snap_col4.metric("Total Charges",    f"${total_charges:.2f}")

# ── Footer / instructions ───────────────────────────────────────────────────────
st.divider()
with st.expander("ℹ️ About this model"):
    st.markdown("""
    **Model:** Random Forest Classifier

    **Pipeline:**
    - SMOTE oversampling to address class imbalance (~73% Not-Churn vs 27% Churn)
    - Label encoding for categorical features
    - StandardScaler for all features
    - Low-importance features dropped: `PhoneService`, `gender`, `StreamingTV`, `StreamingMovies`, `MultipleLines`, `InternetService`

    **Dataset:** IBM Watson Telco Customer Churn — 7,032 customers, 20 features.
    """)
