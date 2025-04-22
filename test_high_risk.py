import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import joblib
from pathlib import Path

def load_and_preprocess_data(data_path=None):
    """
    Load and preprocess the data
    
    Parameters:
    -----------
    data_path : str, optional
        Path to the test data file. If None, uses the UCI heart disease dataset
    """
    if data_path is None:
        # Use default UCI dataset
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
                  'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        df = pd.read_csv(url, names=columns, na_values='?')
    else:
        # Load custom dataset
        df = pd.read_csv(data_path)
    
    # Clean data
    df = df.dropna()
    
    # Convert target to binary
    if 'target' in df.columns:
        df['target'] = df['target'].map(lambda x: 1 if x > 0 else 0)
        X = df.drop('target', axis=1)
        y = df['target']
    else:
        raise ValueError("No 'target' column found in the data")
    
    # Feature engineering
    X['age_squared'] = X['age'] ** 2
    X['trestbps_chol_ratio'] = X['trestbps'] / X['chol']
    X['age_thalach_ratio'] = X['age'] / X['thalach']
    X['cp_thalach'] = X['cp'] * X['thalach']
    X['age_oldpeak'] = X['age'] * X['oldpeak']
    
    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

def preprocess_data(data, scaler):
    """Preprocess the input data for prediction"""
    # Create a copy of the data
    processed_data = data.copy()
    
    # Feature engineering
    processed_data['age_squared'] = processed_data['age'] ** 2
    processed_data['trestbps_chol_ratio'] = processed_data['trestbps'] / processed_data['chol']
    processed_data['age_thalach_ratio'] = processed_data['age'] / processed_data['thalach']
    processed_data['cp_thalach'] = processed_data['cp'] * processed_data['thalach']
    processed_data['age_oldpeak'] = processed_data['age'] * processed_data['oldpeak']
    
    # Select features in the same order as training
    feature_order = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal',
        'age_squared', 'trestbps_chol_ratio', 'age_thalach_ratio',
        'cp_thalach', 'age_oldpeak'
    ]
    
    processed_data = processed_data[feature_order]
    
    # Scale features using the fitted scaler
    scaled_data = scaler.transform(processed_data)
    
    return scaled_data

def get_risk_level(probability):
    """Determine risk level based on probability"""
    if probability < 0.2:
        return "Very Low", "success"
    elif probability < 0.4:
        return "Low", "info"
    elif probability < 0.6:
        return "Moderate", "warning"
    elif probability < 0.8:
        return "High", "warning"
    else:
        return "Very High", "danger"

def get_risk_factors(patient_data):
    """Identify risk factors for a patient"""
    risk_factors = []
    
    if patient_data['age'] > 60:
        risk_factors.append("Advanced age")
    if patient_data['trestbps'] >= 140:
        risk_factors.append("High blood pressure")
    if patient_data['chol'] > 240:
        risk_factors.append("High cholesterol")
    if patient_data['thalach'] < 100:
        risk_factors.append("Low maximum heart rate")
    elif patient_data['thalach'] > 170:
        risk_factors.append("High maximum heart rate")
    if patient_data['fbs'] == 1:
        risk_factors.append("High fasting blood sugar")
    if patient_data['exang'] == 1:
        risk_factors.append("Exercise-induced angina")
    if patient_data['oldpeak'] > 2:
        risk_factors.append("Significant ST depression")
    
    return risk_factors

def main():
    # Load and preprocess UCI dataset to get the scaler
    print("Loading UCI dataset to get the scaler...")
    _, _, scaler = load_and_preprocess_data()
    
    # High-risk patient data
    high_risk_data = {
        'age': 70,  # Advanced age
        'sex': 1,   # Male
        'cp': 2,    # Non-anginal pain
        'trestbps': 180,  # Very high blood pressure
        'chol': 300,  # Very high cholesterol
        'fbs': 1,    # High fasting blood sugar
        'restecg': 2,  # Left ventricular hypertrophy
        'thalach': 100,  # Low maximum heart rate
        'exang': 1,  # Exercise-induced angina
        'oldpeak': 4.0,  # Very significant ST depression
        'slope': 2,  # Downsloping
        'ca': 4,     # 4 major vessels
        'thal': 7    # Reversible defect
    }
    
    # Create DataFrame
    input_data = pd.DataFrame([high_risk_data])
    
    # Load model
    model_path = Path("models/grid_search_best_model.joblib")
    if not model_path.exists():
        print("Model file not found. Please ensure the model is properly installed.")
        return
    
    model = joblib.load(model_path)
    if model is None:
        print("Error loading the model. Please try again later.")
        return
    
    # Preprocess data using the fitted scaler
    preprocessed_data = preprocess_data(input_data, scaler)
    
    # Make prediction
    prediction = model.predict(preprocessed_data)[0]
    probability = model.predict_proba(preprocessed_data)[0][1]
    
    # Get risk assessment
    risk_level, risk_class = get_risk_level(probability)
    risk_factors = get_risk_factors(input_data.iloc[0])
    
    # Print results
    print("\nHigh-Risk Patient Assessment")
    print("=" * 30)
    print(f"Prediction: {'Heart Disease' if prediction == 1 else 'No Heart Disease'}")
    print(f"Probability: {probability:.2%}")
    print(f"Risk Level: {risk_level}")
    print("\nRisk Factors:")
    for factor in risk_factors:
        print(f"- {factor}")
    print("\nPatient Data:")
    for key, value in high_risk_data.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main() 