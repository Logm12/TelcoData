import pandas as pd
import numpy as np
import joblib
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from typing import Dict, Any

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.preprocessing import DataPreprocessor
from src.utils import get_logger

logger = get_logger(__name__)

MODELS_DIR = "models"
DATA_PATH = "data/raw/telecom_churn.csv"

def train():
    logger.info("Starting training pipeline...")
    
    # 1. Load Data
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data(DATA_PATH)
    
    # 2. Clean Data
    df_clean = preprocessor.clean_data(df)
    
    # 3. Split Data
    X_train_raw, X_test_raw, y_train, y_test = preprocessor.split_data(df_clean)
    
    # 4. Preprocess (Fit on Train, Transform Test)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    
    # 5. Handle Imbalance (SMOTE)
    logger.info("Applying SMOTE to handle class imbalance...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    logger.info(f"Original train shape: {y_train.shape}, Resampled train shape: {y_train_resampled.shape}")
    
    # 6. Train Models
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)
    }
    
    best_model = None
    best_auc = 0
    best_model_name = ""
    
    results = {}
    
    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train_resampled, y_train_resampled)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y_test, y_prob)
        acc = accuracy_score(y_test, y_pred)
        
        logger.info(f"{name} Results: Accuracy={acc:.4f}, AUC={auc:.4f}")
        logger.info("\n" + classification_report(y_test, y_pred))
        
        results[name] = {"model": model, "auc": auc}
        
        if auc > best_auc:
            best_auc = auc
            best_model = model
            best_model_name = name
            
    logger.info(f"Best model: {best_model_name} with AUC: {best_auc:.4f}")
    
    # 7. Save Artifacts
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Save the best model
    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    joblib.dump(best_model, model_path)
    logger.info(f"Best model saved to {model_path}")

    # Save the preprocessor (which contains the fitted scaling/encoding)
    preprocessor_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_path)
    logger.info(f"Preprocessor saved to {preprocessor_path}")
    
    # Save feature names for SHAP
    feature_names = preprocessor.get_feature_names()
    feature_names_path = os.path.join(MODELS_DIR, "feature_names.joblib")
    joblib.dump(feature_names, feature_names_path)
    
    logger.info("Training pipeline completed successfully.")

if __name__ == "__main__":
    train()
