import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from typing import Tuple, List, Optional
import sys
import os

# Add src to path to import utils if run as script (optional but good practice)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import get_logger

logger = get_logger(__name__)

class DataPreprocessor:
    """
    A class to handle data loading, cleaning, and preprocessing for Telco Churn prediction.
    """
    
    def __init__(self):
        self.numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        self.categorical_features = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 
            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
            'Contract', 'PaperlessBilling', 'PaymentMethod'
        ]
        self.preprocessor = None
        self.feature_names = None

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Loads data from a CSV file.

        Args:
            filepath (str): Path to the CSV file.

        Returns:
            pd.DataFrame: Loaded dataframe.
        """
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Data loaded successfully from {filepath}, shape: {df.shape}")
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs initial data cleaning:
        1. Drop customerID (not useful for prediction).
        2. Convert TotalCharges to numeric (handling empty strings/errors).
        3. Drop rows where Churn is missing (if any).
        
        Args:
            df (pd.DataFrame): Raw dataframe.
            
        Returns:
            pd.DataFrame: Cleaned dataframe.
        """
        df = df.copy()
        
        # Drop ID
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
            
        # Convert TotalCharges to numeric, coerce errors to NaN
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        # Map target variable
        if 'Churn' in df.columns:
            df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
            
        return df

    def split_data(self, df: pd.DataFrame, target_col: str = 'Churn', test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Splits data into train and test sets.
        
        Args:
            df (pd.DataFrame): The dataframe.
            target_col (str): The name of the target column.
            test_size (float): Proportion of the dataset to include in the test split.
            
        Returns:
            Tuple[X_train, X_test, y_train, y_test]
        """
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        logger.info(f"Splitting data with test_size={test_size}")
        return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        """
        Fits the preprocessor on training data and transforms it.
        
        Args:
            X_train (pd.DataFrame): Training features.
            
        Returns:
            np.ndarray: Processed training features.
        """
        logger.info("Fitting preprocessor...")
        
        # Numeric pipeline: Impute missing (median) -> Scale
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Categorical pipeline: Impute missing (frequent) -> OneHot
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # sparse_output=False for easy DataFrame conversion if needed
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ]
        )
        
        X_processed = self.preprocessor.fit_transform(X_train)
        
        # Capture feature names
        num_features = self.numeric_features
        cat_features = self.preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(self.categorical_features).tolist()
        self.feature_names = num_features + cat_features
        
        logger.info(f"Preprocessing complete. Number of features: {len(self.feature_names)}")
        return X_processed

    def transform(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Transforms new data using the fitted preprocessor.
        
        Args:
            X_test (pd.DataFrame): New data to transform.
            
        Returns:
            np.ndarray: Processed data.
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor has not been fitted yet.")
        return self.preprocessor.transform(X_test)

    def get_feature_names(self) -> List[str]:
        """Returns list of feature names after preprocessing."""
        return self.feature_names
