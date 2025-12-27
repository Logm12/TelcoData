import streamlit as st
import joblib
import os
import sys

# Ensure project root is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.joblib")
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "feature_names.joblib")

@st.cache_resource
def load_artifacts():
    """Load model, preprocessor and feature names."""
    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        feature_names = joblib.load(FEATURE_NAMES_PATH)
        return model, preprocessor, feature_names
    except FileNotFoundError:
        st.error("Model artifacts not found. Please run the training script first.")
        return None, None, None

def get_translated_feature_name(feature_name, lang, mapping):
    """
    Translates a feature name using the mapping.
    Also tries to handle 'Feature_Value' patterns if exact match fails.
    """
    if lang == "en":
        return feature_name
    
    # Direct match
    if feature_name in mapping["vi"]:
        return mapping["vi"][feature_name]
    
    # Attempt to handle OneHot features (e.g., InternetService_DSL -> Internet: DSL)
    # This assumes we captured most in the static mapping, but good to have fallback if needed.
    return feature_name

def render_sidebar_controls(t):
    """
    Renders sidebar controls and returns a dictionary of inputs.
    """
    st.sidebar.header(t["sidebar_header"])
    
    def binary_select(key, label, options):
        # We assume options are passed in English values (for the model)
        # but we could map them for display if we wanted to go deeper.
        # For now, let's just translate the Label.
        return st.sidebar.selectbox(label, options, key=key)

    # Note: The model expects specific English string values (e.g. 'Yes', 'No', 'Male').
    # To fully translate the dropdown OPTIONs, we would need a map from VI Display -> EN Value.
    # For simplicity in this iteration, we keep the OPTION values in English/Data format
    # but translate the LABEL. The user is technical enough (Data Scientist portfolio).
    
    # Categorical features
    gender = binary_select("gender", t["lbl_gender"], ("Male", "Female"))
    senior_citizen = binary_select("senior", t["lbl_senior"], (0, 1))
    partner = binary_select("partner", t["lbl_partner"], ("Yes", "No"))
    dependents = binary_select("dependents", t["lbl_dependents"], ("Yes", "No"))
    phone_service = binary_select("phone", t["lbl_phone"], ("Yes", "No"))
    multiple_lines = binary_select("multiline", t["lbl_files"], ("Yes", "No", "No phone service"))
    internet_service = binary_select("internet", t["lbl_internet"], ("DSL", "Fiber optic", "No"))
    online_security = binary_select("security", t["lbl_security"], ("Yes", "No", "No internet service"))
    online_backup = binary_select("backup", t["lbl_backup"], ("Yes", "No", "No internet service"))
    device_protection = binary_select("device", t["lbl_device"], ("Yes", "No", "No internet service"))
    tech_support = binary_select("tech", t["lbl_tech"], ("Yes", "No", "No internet service"))
    streaming_tv = binary_select("tv", t["lbl_tv"], ("Yes", "No", "No internet service"))
    streaming_movies = binary_select("movies", t["lbl_movies"], ("Yes", "No", "No internet service"))
    contract = binary_select("contract", t["lbl_contract"], ("Month-to-month", "One year", "Two year"))
    paperless_billing = binary_select("paperless", t["lbl_paperless"], ("Yes", "No"))
    payment_method = binary_select("payment", t["lbl_payment"], (
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ))

    # Numerical features
    tenure = st.sidebar.slider(t["lbl_tenure"], 0, 72, 12, key="tenure")
    monthly_charges = st.sidebar.number_input(t["lbl_monthly"], 18.25, 118.75, 50.0, key="monthly")
    total_charges = st.sidebar.number_input(t["lbl_total"], 0.0, 9000.0, monthly_charges * tenure, key="total")

    data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    return data
