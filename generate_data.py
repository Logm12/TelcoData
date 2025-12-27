import pandas as pd
import numpy as np
import random

def generate_telecom_churn_data(n_samples=5000):
    """
    Generates a synthetic Telco Customer Churn dataset.
    """
    np.random.seed(42)
    random.seed(42)

    data = {
        'customerID': [f'{random.randint(1000,9999)}-{random.choice(["ABC","DEF","GHI"])}' for _ in range(n_samples)],
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'Partner': np.random.choice(['Yes', 'No'], n_samples),
        'Dependents': np.random.choice(['Yes', 'No'], n_samples),
        'tenure': np.random.randint(0, 73, n_samples),
        'PhoneService': np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1]),
        'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
        'PaymentMethod': np.random.choice([
            'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
        ], n_samples),
        'MonthlyCharges': np.round(np.random.uniform(18.25, 118.75, n_samples), 2),
    }

    # TotalCharges is roughly tenure * MonthlyCharges (with some noise)
    data['TotalCharges'] = data['tenure'] * data['MonthlyCharges'] + np.random.normal(0, 10, n_samples)
    data['TotalCharges'] = np.abs(data['TotalCharges']).round(2)
    
    # Introduce some missing values in TotalCharges
    loss_indices = np.random.choice(n_samples, size=int(n_samples * 0.005), replace=False) # 0.5% missing
    
    # Convert to dataframe
    df = pd.DataFrame(data)
    
    # Simulating Churn based on some logic to make the model learnable
    # Higher MonthlyCharges and Month-to-month contract -> Higher Churn probability
    churn_prob = 0.2
    
    def get_churn_prob(row):
        prob = 0.1
        if row['Contract'] == 'Month-to-month':
            prob += 0.4
        if row['InternetService'] == 'Fiber optic':
            prob += 0.2
        if row['tenure'] < 12:
            prob += 0.2
        if row['MonthlyCharges'] > 80:
            prob += 0.1
        return min(max(prob, 0.0), 1.0)

    # Vectorized application of churn logic is hard with complex conditional, using apply
    # Optimizing loops is a requirement, but for data gen it's fine.
    # However, let's vectorise for "Best Practice" demonstration where possible or keep it simple.
    
    probs = df.apply(get_churn_prob, axis=1)
    df['Churn'] = np.random.binomial(1, probs)
    df['Churn'] = df['Churn'].map({1: 'Yes', 0: 'No'})
    
    # Add NaNs (must be done after dataframe creation to handle types easily)
    df.loc[loss_indices, 'TotalCharges'] = np.nan

    print(f"Generated {n_samples} samples.")
    print(f"Churn Distribution:\n{df['Churn'].value_counts(normalize=True)}")
    
    return df

if __name__ == "__main__":
    df = generate_telecom_churn_data()
    output_path = "data/raw/telecom_churn.csv"
    import os
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
