from google.colab import drive
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

def get_risk_level(probability):
    """Determine risk level based on probability"""
    if probability < 0.2:
        return "Very Low"
    elif probability < 0.4:
        return "Low"
    elif probability < 0.6:
        return "Moderate"
    elif probability < 0.8:
        return "High"
    else:
        return "Very High"

def get_risk_factors(patient_data):
    """Identify risk factors for a patient"""
    risk_factors = []
    
    # Age risk
    if patient_data['age'] > 60:
        risk_factors.append("Advanced age")
    
    # Blood pressure risk
    if patient_data['trestbps'] >= 140:
        risk_factors.append("High blood pressure")
    
    # Cholesterol risk
    if patient_data['chol'] > 240:
        risk_factors.append("High cholesterol")
    
    # Heart rate risk
    if patient_data['thalach'] < 100:
        risk_factors.append("Low maximum heart rate")
    elif patient_data['thalach'] > 170:
        risk_factors.append("High maximum heart rate")
    
    # Other factors
    if patient_data['fbs'] == 1:
        risk_factors.append("High fasting blood sugar")
    if patient_data['exang'] == 1:
        risk_factors.append("Exercise-induced angina")
    if patient_data['oldpeak'] > 2:
        risk_factors.append("Significant ST depression")
    
    return risk_factors

def get_recommendations(risk_factors, risk_level):
    """Generate recommendations based on risk factors and level"""
    recommendations = []
    
    if "High blood pressure" in risk_factors:
        recommendations.append("- Regular blood pressure monitoring")
        recommendations.append("- Consider reducing sodium intake")
    
    if "High cholesterol" in risk_factors:
        recommendations.append("- Regular cholesterol screening")
        recommendations.append("- Consider diet modifications")
    
    if "High fasting blood sugar" in risk_factors:
        recommendations.append("- Regular blood sugar monitoring")
        recommendations.append("- Consider consulting with a diabetes specialist")
    
    if risk_level in ["High", "Very High"]:
        recommendations.append("- Immediate consultation with a cardiologist")
        recommendations.append("- Consider stress test and detailed heart examination")
    
    if not recommendations:
        recommendations.append("- Maintain current healthy lifestyle")
        recommendations.append("- Regular check-ups as recommended by your doctor")
    
    return recommendations

def preprocess_data(data):
    """Preprocess the input data for prediction"""
    # Feature engineering
    data['age_squared'] = data['age'] ** 2
    data['trestbps_chol_ratio'] = data['trestbps'] / data['chol']
    data['age_thalach_ratio'] = data['age'] / data['thalach']
    data['cp_thalach'] = data['cp'] * data['thalach']
    data['age_oldpeak'] = data['age'] * data['oldpeak']
    
    # Scale features
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(data)
    
    return scaled_data

def main():
    # Mount Google Drive
    drive.mount('/content/drive', force_remount=True)
    
    # Load the best model
    model_path = '/content/drive/MyDrive/grid_search_best_model.joblib'
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    # Load test data
    test_data_path = '/content/drive/MyDrive/heart_test_data.csv'
    try:
        test_data = pd.read_csv(test_data_path)
        print(f"Test data loaded successfully from {test_data_path}")
        print(f"Number of patients: {len(test_data)}")
    except Exception as e:
        print(f"Error loading test data: {str(e)}")
        return
    
    # Preprocess data
    X_test = preprocess_data(test_data)
    
    # Make predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    # Save results to file
    output_path = '/content/drive/MyDrive/heart_disease_risk_assessment_results.txt'
    with open(output_path, 'w') as f:
        f.write("Heart Disease Risk Assessment Results\n")
        f.write("=" * 70 + "\n\n")
        
        for i, (pred, prob) in enumerate(zip(predictions, probabilities[:, 1])):
            patient_data = test_data.iloc[i]
            risk_level = get_risk_level(prob)
            risk_factors = get_risk_factors(patient_data)
            recommendations = get_recommendations(risk_factors, risk_level)
            
            f.write(f"Patient {i+1}:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Age: {patient_data['age']}\n")
            f.write(f"Gender: {'Male' if patient_data['sex'] == 1 else 'Female'}\n")
            f.write(f"Prediction: {'Heart Disease' if pred == 1 else 'No Heart Disease'}\n")
            f.write(f"Risk Level: {risk_level} ({prob:.1%} probability)\n\n")
            
            if risk_factors:
                f.write("Identified Risk Factors:\n")
                for factor in risk_factors:
                    f.write(f"• {factor}\n")
                f.write("\n")
            else:
                f.write("No significant risk factors identified\n\n")
            
            f.write("Recommendations:\n")
            for rec in recommendations:
                f.write(f"{rec}\n")
            f.write("\n" + "=" * 70 + "\n\n")
    
    # Display enhanced results
    print("\nDetailed Analysis Results:")
    print("=" * 70)
    for i, (pred, prob) in enumerate(zip(predictions, probabilities[:, 1])):
        patient_data = test_data.iloc[i]
        risk_level = get_risk_level(prob)
        risk_factors = get_risk_factors(patient_data)
        recommendations = get_recommendations(risk_factors, risk_level)
        
        print(f"\nPatient {i+1}:")
        print("-" * 40)
        print(f"Age: {patient_data['age']}")
        print(f"Gender: {'Male' if patient_data['sex'] == 1 else 'Female'}")
        print(f"Prediction: {'Heart Disease' if pred == 1 else 'No Heart Disease'}")
        print(f"Risk Level: {risk_level} ({prob:.1%} probability)")
        
        if risk_factors:
            print("\nIdentified Risk Factors:")
            for factor in risk_factors:
                print(f"• {factor}")
        else:
            print("\nNo significant risk factors identified")
        
        print("\nRecommendations:")
        for rec in recommendations:
            print(rec)

    print(f"\nDetailed results saved to: {output_path}")

if __name__ == "__main__":
    main() 