import streamlit as st
import joblib
import pandas as pd

# ── Load model artifacts ───────────────────────────────────────────────────────
model         = joblib.load("churn_model.pkl")
scaler        = joblib.load("churn_scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RedFlag AI",
    page_icon="🚩",
    layout="centered"
)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🚩 RedFlag AI")
st.markdown("Identify customers at risk of leaving — **before it's too late.**")
st.caption("Model accuracy: 81.55% | Algorithm: Logistic Regression | Dataset: Telco Customer Churn")
st.divider()

# ── Input Form ─────────────────────────────────────────────────────────────────
st.subheader("Customer Details")

col1, col2 = st.columns(2)

with col1:
    gender         = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner        = st.selectbox("Has Partner", ["No", "Yes"])
    dependents     = st.selectbox("Has Dependents", ["No", "Yes"])
    tenure         = st.slider("Tenure (months)", 0, 72, 12)
    phone_service  = st.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

with col2:
    internet_service  = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security   = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup     = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support      = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv      = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies  = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

st.divider()
st.subheader("Billing Details")

col3, col4 = st.columns(2)

with col3:
    contract       = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless      = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

with col4:
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=65.0)
    total_charges   = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=1500.0)

st.divider()

# ── Encode inputs ──────────────────────────────────────────────────────────────
def encode_inputs():
    data = {
        "gender":            1 if gender == "Male" else 0,
        "SeniorCitizen":     1 if senior_citizen == "Yes" else 0,
        "Partner":           1 if partner == "Yes" else 0,
        "Dependents":        1 if dependents == "Yes" else 0,
        "tenure":            tenure,
        "PhoneService":      1 if phone_service == "Yes" else 0,
        "MultipleLines":     ["No phone service", "No", "Yes"].index(multiple_lines),
        "InternetService":   ["DSL", "Fiber optic", "No"].index(internet_service),
        "OnlineSecurity":    ["No internet service", "No", "Yes"].index(online_security),
        "OnlineBackup":      ["No internet service", "No", "Yes"].index(online_backup),
        "DeviceProtection":  ["No internet service", "No", "Yes"].index(device_protection),
        "TechSupport":       ["No internet service", "No", "Yes"].index(tech_support),
        "StreamingTV":       ["No internet service", "No", "Yes"].index(streaming_tv),
        "StreamingMovies":   ["No internet service", "No", "Yes"].index(streaming_movies),
        "Contract":          ["Month-to-month", "One year", "Two year"].index(contract),
        "PaperlessBilling":  1 if paperless == "Yes" else 0,
        "PaymentMethod":     [
                                "Bank transfer (automatic)",
                                "Credit card (automatic)",
                                "Electronic check",
                                "Mailed check"
                             ].index(payment_method),
        "MonthlyCharges":    monthly_charges,
        "TotalCharges":      total_charges,
    }
    return pd.DataFrame([data])[feature_names]

# ── Predict ────────────────────────────────────────────────────────────────────
if st.button("🔍 Analyze Customer", use_container_width=True):
    input_df     = encode_inputs()
    input_scaled = scaler.transform(input_df)
    prediction   = model.predict(input_scaled)[0]
    probability  = model.predict_proba(input_scaled)[0][1]

    st.divider()
    st.subheader("Analysis Result")

    if prediction == 1:
        st.error(f"🚩 **High Churn Risk** — This customer has a {probability:.1%} probability of leaving.")
        st.markdown("**Recommended Actions:**")
        st.markdown("- Offer a discounted long-term contract")
        st.markdown("- Assign a dedicated customer success manager")
        st.markdown("- Provide a loyalty reward or free upgrade")
    else:
        st.success(f"✅ **Low Churn Risk** — This customer has a {1 - probability:.1%} probability of staying.")
        st.markdown("**Recommended Actions:**")
        st.markdown("- Continue current engagement strategy")
        st.markdown("- Consider upselling premium services")
        st.markdown("- Enroll in loyalty rewards program")

    st.divider()
    st.markdown("**Churn Risk Score**")
    st.progress(float(probability))
    st.caption(f"Risk level: {probability:.1%}")