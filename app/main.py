import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from app.translations import UI_TEXT, FEATURE_MAPPING
from app.app_utils import load_artifacts, render_sidebar_controls, get_translated_feature_name

st.set_page_config(
    page_title="Telco Churn Prediction",
    page_icon="📡",
    layout="wide"
)

def main():
    # 1. Language Selection
    lang_choice = st.sidebar.selectbox("Language / Ngôn Ngữ", ["English", "Tiếng Việt"])
    lang = "vi" if lang_choice == "Tiếng Việt" else "en"
    t = UI_TEXT[lang]

    # 2. Header
    st.title(t["title"])
    st.markdown(t["subtitle"])

    # 3. Load artifacts
    model, preprocessor, features_raw = load_artifacts()
    if model is None:
        return

    # 4. User Inputs
    # render_sidebar_controls now returns a raw dictionary
    input_data = render_sidebar_controls(t)
    input_df = pd.DataFrame(input_data, index=[0])
    
    with st.expander(t["view_input"]):
        st.dataframe(input_df)

    # 5. Prediction Logic
    if st.button(t["predict_btn"]):
        try:
            # Transform
            processed_data = preprocessor.transform(input_df)
            
            # Predict
            prediction = model.predict(processed_data)[0]
            probability = model.predict_proba(processed_data)[0][1]
            
            # Display Result
            st.divider()
            st.subheader(t["results_header"])
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error(t["churn_yes"])
                else:
                    st.success(t["churn_no"])
            
            with col2:
                st.metric(t["probability_label"], f"{probability:.2%}")

            # 6. SHAP Explanation
            st.divider()
            st.subheader(t["shap_header"])
            
            with st.spinner(t["calc_shap"]):
                explainer = shap.TreeExplainer(model)
                shap_explanation = explainer(processed_data)
                
                # Handle SHAP dimensionality
                if len(shap_explanation.shape) == 3:
                    explanation_to_plot = shap_explanation[0, :, 1]
                else:
                    explanation_to_plot = shap_explanation[0]
                
                # TRANSLATE FEATURE NAMES FOR PLOT
                # We map the raw feature names (from model) to display names (VI/EN)
                translated_features = [
                    get_translated_feature_name(f, lang, FEATURE_MAPPING) for f in features_raw
                ]
                explanation_to_plot.feature_names = translated_features
                
                # Graph
                fig, ax = plt.subplots(figsize=(10, 8))
                shap.plots.waterfall(explanation_to_plot, show=False, max_display=10)
                st.pyplot(fig)
                
                # Text Interpretation
                display_text_explanation(explanation_to_plot, translated_features, t)

            # 7. Recommendations
            st.divider()
            if probability > 0.5:
                st.warning(t["risk_high"])
                st.markdown(t["rec_discount"])
                st.markdown(t["rec_tech"])
                st.markdown(t["rec_upgrade"])
            else:
                st.success(t["risk_low"])
                
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            import traceback
            st.code(traceback.format_exc())

def display_text_explanation(shap_obj, feature_names, t):
    """
    Displays text analysis of the SHAP values using the translated feature names.
    """
    values = shap_obj.values
    
    # Zip names and values
    feats = list(zip(feature_names, values))
    
    # Sort
    pos_features = sorted([f for f in feats if f[1] > 0], key=lambda x: x[1], reverse=True)
    neg_features = sorted([f for f in feats if f[1] < 0], key=lambda x: x[1])
    
    st.markdown(f"### {t['top_features']}")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"#### 📈 {t['factor_increase']}")
        if pos_features:
            for feat, val in pos_features[:3]:
                st.write(f"- **{feat}**: (+{val:.3f})")
        else:
            st.write("...")
            
    with col2:
        st.markdown(f"#### 📉 {t['factor_decrease']}")
        if neg_features:
            for feat, val in neg_features[:3]:
                st.write(f"- **{feat}**: ({val:.3f})")
        else:
            st.write("...")

if __name__ == "__main__":
    main()
