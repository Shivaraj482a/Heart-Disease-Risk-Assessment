import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from google.colab import drive
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler

# Set page configuration
st.set_page_config(
    page_title="Heart Disease Risk Assessment",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #e74c3c;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3498db;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .info-text {
        font-size: 1.1rem;
        color: #2c3e50;
    }
    .result-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .high-risk {
        color: #e74c3c;
        font-weight: bold;
    }
    .moderate-risk {
        color: #f39c12;
        font-weight: bold;
    }
    .low-risk {
        color: #27ae60;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 1.1rem;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
</style>
""", unsafe_allow_html=True)

# Function to load the model
@st.cache_resource
def load_model():
    try:
        # For local testing without Google Drive
        if os.path.exists('grid_search_best_model.joblib'):
            return joblib.load('grid_search_best_model.joblib')
        
        # For Google Colab with Google Drive
        drive.mount('/content/drive', force_remount=True)
        model_path = '/content/drive/MyDrive/grid_search_best_model.joblib'
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Function to preprocess input data
def preprocess_data(data):
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

# Function to determine risk level
def get_risk_level(probability):
    if probability < 0.2:
        return "Very Low", "low-risk"
    elif probability < 0.4:
        return "Low", "low-risk"
    elif probability < 0.6:
        return "Moderate", "moderate-risk"
    elif probability < 0.8:
        return "High", "high-risk"
    else:
        return "Very High", "high-risk"

# Function to identify risk factors
def get_risk_factors(patient_data):
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

# Function to generate recommendations
def get_recommendations(risk_factors, risk_level):
    recommendations = []
    
    if "High blood pressure" in risk_factors:
        recommendations.append("Regular blood pressure monitoring")
        recommendations.append("Consider reducing sodium intake")
    
    if "High cholesterol" in risk_factors:
        recommendations.append("Regular cholesterol screening")
        recommendations.append("Consider diet modifications")
    
    if "High fasting blood sugar" in risk_factors:
        recommendations.append("Regular blood sugar monitoring")
        recommendations.append("Consider consulting with a diabetes specialist")
    
    if risk_level in ["High", "Very High"]:
        recommendations.append("Immediate consultation with a cardiologist")
        recommendations.append("Consider stress test and detailed heart examination")
    
    if not recommendations:
        recommendations.append("Maintain current healthy lifestyle")
        recommendations.append("Regular check-ups as recommended by your doctor")
    
    return recommendations

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">❤️ Heart Disease Risk Assessment</h1>', unsafe_allow_html=True)
    
    # Sidebar with information
    with st.sidebar:
        st.markdown("## About This App")
        st.markdown("""
        This application uses machine learning to assess your risk of heart disease based on medical data.
        
        **How it works:**
        1. Enter your medical information
        2. Click 'Assess Risk'
        3. Receive your personalized risk assessment
        
        **Note:** This tool is for informational purposes only and should not replace professional medical advice.
        """)
        
        st.markdown("## Data Privacy")
        st.markdown("""
        Your data is processed locally and is not stored or transmitted to any server.
        """)
        
        st.markdown("## Feature Descriptions")
        st.markdown("""
        - **Age:** Age in years
        - **Sex:** 1 = male; 0 = female
        - **Chest Pain Type:** 0-3 (0: typical angina, 1: atypical angina, 2: non-anginal pain, 3: asymptomatic)
        - **Resting BP:** Resting blood pressure (mm Hg)
        - **Cholesterol:** Serum cholesterol (mg/dl)
        - **Fasting Blood Sugar:** > 120 mg/dl (1 = true; 0 = false)
        - **Resting ECG:** 0 = normal, 1 = ST-T wave abnormality, 2 = left ventricular hypertrophy
        - **Max Heart Rate:** Maximum heart rate achieved
        - **Exercise Angina:** Exercise induced angina (1 = yes; 0 = no)
        - **ST Depression:** ST depression induced by exercise relative to rest
        - **Slope:** Slope of the peak exercise ST segment (0 = upsloping, 1 = flat, 2 = downsloping)
        - **Vessels:** Number of major vessels (0-3) colored by fluoroscopy
        - **Thalassemia:** 0 = normal; 1 = fixed defect; 2 = reversable defect
        """)
    
    # Main content
    st.markdown('<h2 class="sub-header">Enter Your Medical Information</h2>', unsafe_allow_html=True)
    
    # Create two columns for input fields
    col1, col2 = st.columns(2)
    
    with col1:
        # Personal information
        st.markdown("### Personal Information")
        age = st.slider("Age", 20, 100, 50)
        sex = st.radio("Sex", ["Male", "Female"])
        sex_value = 1 if sex == "Male" else 0
        
        # Chest pain and blood pressure
        st.markdown("### Heart-Related Information")
        cp = st.selectbox("Chest Pain Type", 
                         ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"],
                         index=0)
        cp_value = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp)
        
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
        chol = st.slider("Cholesterol (mg/dl)", 100, 600, 200)
        
        # Blood sugar and ECG
        fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
        fbs_value = 1 if fbs == "Yes" else 0
        
        restecg = st.selectbox("Resting ECG Results", 
                              ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"],
                              index=0)
        restecg_value = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(restecg)
    
    with col2:
        # Heart rate and exercise
        st.markdown("### Exercise Information")
        thalach = st.slider("Maximum Heart Rate Achieved", 60, 202, 150)
        
        exang = st.radio("Exercise-Induced Angina", ["No", "Yes"])
        exang_value = 1 if exang == "Yes" else 0
        
        oldpeak = st.slider("ST Depression", 0.0, 6.0, 0.0, 0.1)
        
        # Additional heart information
        st.markdown("### Additional Heart Information")
        slope = st.selectbox("Slope of Peak Exercise ST", 
                            ["Upsloping", "Flat", "Downsloping"],
                            index=0)
        slope_value = ["Upsloping", "Flat", "Downsloping"].index(slope)
        
        ca = st.slider("Number of Major Vessels (0-3)", 0, 3, 0)
        
        thal = st.selectbox("Thalassemia", 
                           ["Normal", "Fixed Defect", "Reversable Defect"],
                           index=0)
        thal_value = ["Normal", "Fixed Defect", "Reversable Defect"].index(thal)
    
    # Create a button to assess risk
    if st.button("Assess Heart Disease Risk"):
        # Load the model
        model = load_model()
        
        if model is not None:
            # Create a DataFrame with the input data
            input_data = pd.DataFrame({
                'age': [age],
                'sex': [sex_value],
                'cp': [cp_value],
                'trestbps': [trestbps],
                'chol': [chol],
                'fbs': [fbs_value],
                'restecg': [restecg_value],
                'thalach': [thalach],
                'exang': [exang_value],
                'oldpeak': [oldpeak],
                'slope': [slope_value],
                'ca': [ca],
                'thal': [thal_value]
            })
            
            # Preprocess the data
            processed_data = preprocess_data(input_data)
            
            # Make prediction
            prediction = model.predict(processed_data)[0]
            probability = model.predict_proba(processed_data)[0][1]
            
            # Get risk level and factors
            risk_level, risk_class = get_risk_level(probability)
            risk_factors = get_risk_factors(input_data.iloc[0])
            recommendations = get_recommendations(risk_factors, risk_level)
            
            # Display results
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            
            # Prediction result
            st.markdown("## Assessment Results")
            
            # Create a progress bar for risk level
            st.markdown(f"### Risk Level: <span class='{risk_class}'>{risk_level}</span>", unsafe_allow_html=True)
            st.progress(probability)
            st.markdown(f"**Probability of Heart Disease:** {probability:.1%}")
            
            # Prediction
            if prediction == 1:
                st.markdown("### Diagnosis: <span class='high-risk'>Heart Disease Detected</span>", unsafe_allow_html=True)
            else:
                st.markdown("### Diagnosis: <span class='low-risk'>No Heart Disease Detected</span>", unsafe_allow_html=True)
            
            # Risk factors
            if risk_factors:
                st.markdown("### Identified Risk Factors:")
                for factor in risk_factors:
                    st.markdown(f"- {factor}")
            else:
                st.markdown("### No significant risk factors identified")
            
            # Recommendations
            st.markdown("### Recommendations:")
            for rec in recommendations:
                st.markdown(f"- {rec}")
            
            # Visualization
            st.markdown("### Risk Visualization")
            
            # Create a gauge chart
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.set_style("whitegrid")
            
            # Create a color gradient based on risk level
            if risk_class == "low-risk":
                color = "green"
            elif risk_class == "moderate-risk":
                color = "orange"
            else:
                color = "red"
            
            # Create a horizontal bar chart
            ax.barh([0], [probability], color=color, height=0.3)
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.5, 0.5)
            ax.set_yticks([])
            ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_xticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
            ax.set_title(f"Heart Disease Risk: {risk_level}")
            
            # Add a vertical line at the current probability
            ax.axvline(x=probability, color='black', linestyle='--', alpha=0.7)
            
            st.pyplot(fig)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Disclaimer
            st.markdown("""
            <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 20px;">
                <p><strong>Disclaimer:</strong> This assessment is for informational purposes only and should not replace professional medical advice. 
                Always consult with a healthcare provider for proper diagnosis and treatment.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("Unable to load the prediction model. Please try again later.")

if __name__ == "__main__":
    main() 