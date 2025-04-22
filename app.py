from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_file
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import joblib
from pathlib import Path
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from models import db, User, Assessment
from forms import LoginForm, RegistrationForm
import json
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from io import BytesIO
from datetime import datetime
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24).hex()  # Generate a secure random key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///heart_disease.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Create database tables
with app.app_context():
    db.create_all()

@app.template_filter('fromjson')
def fromjson_filter(value):
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []

@login_manager.user_loader
def load_user(id):
    return db.session.get(User, int(id))

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

def get_recommendations(risk_factors, risk_level):
    """Generate recommendations based on risk factors and level"""
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

def load_and_preprocess_data():
    """Load UCI dataset and get scaler"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
              'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    df = pd.read_csv(url, names=columns, na_values='?')
    df = df.dropna()
    
    # Convert target to binary
    df['target'] = df['target'].map(lambda x: 1 if x > 0 else 0)
    X = df.drop('target', axis=1)
    
    # Feature engineering on training data
    X['age_squared'] = X['age'] ** 2
    X['trestbps_chol_ratio'] = X['trestbps'] / X['chol']
    X['age_thalach_ratio'] = X['age'] / X['thalach']
    X['cp_thalach'] = X['cp'] * X['thalach']
    X['age_oldpeak'] = X['age'] * X['oldpeak']
    
    # Create and fit scaler on training data
    scaler = RobustScaler()
    scaler.fit(X)
    return scaler

def preprocess_data(data):
    """Preprocess the input data for prediction"""
    print("\nPreprocessing steps:")
    print("1. Input data:")
    print(data)
    
    # Create a copy of the data
    processed_data = data.copy()
    
    # Feature engineering
    print("\n2. Performing feature engineering...")
    processed_data['age_squared'] = processed_data['age'] ** 2
    processed_data['trestbps_chol_ratio'] = processed_data['trestbps'] / processed_data['chol']
    processed_data['age_thalach_ratio'] = processed_data['age'] / processed_data['thalach']
    processed_data['cp_thalach'] = processed_data['cp'] * processed_data['thalach']
    processed_data['age_oldpeak'] = processed_data['age'] * processed_data['oldpeak']
    
    print("Feature engineering results:")
    print(processed_data)
    
    # Select features in the same order as training
    feature_order = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal',
        'age_squared', 'trestbps_chol_ratio', 'age_thalach_ratio',
        'cp_thalach', 'age_oldpeak'
    ]
    
    print("\n3. Selecting and ordering features...")
    processed_data = processed_data[feature_order]
    print("Features after ordering:")
    print(processed_data.columns.tolist())
    
    # Get scaler fitted on UCI dataset
    print("\n4. Scaling data...")
    scaler = load_and_preprocess_data()
    scaled_data = scaler.transform(processed_data)
    print("Data scaled successfully")
    print("Scaled data shape:", scaled_data.shape)
    
    return scaled_data

# Load the model
def load_model():
    try:
        model_path = Path("models/grid_search_best_model.joblib")
        if not model_path.exists():
            raise FileNotFoundError("Model file not found")
        return joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

# Global variable for model
model = load_model()

# Authentication routes
@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and user.check_password(form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('dashboard'))
        flash('Invalid email or password', 'error')
    return render_template('login.html', form=form)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    form = RegistrationForm()
    if form.validate_on_submit():
        if User.query.filter_by(email=form.email.data).first():
            flash('Email already exists', 'error')
            return render_template('signup.html', form=form)
        
        user = User(name=form.name.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Registration successful!', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/dashboard')
@login_required
def dashboard():
    assessments = Assessment.query.filter_by(user_id=current_user.id).order_by(Assessment.date.desc()).all()
    
    # Calculate statistics for dashboard
    total_assessments = len(assessments)
    if total_assessments > 0:
        latest_assessment = assessments[0]
        current_risk_level = latest_assessment.risk_level
        current_risk_class = "low" if current_risk_level == "Low" else "moderate" if current_risk_level == "Moderate" else "high"
        latest_assessment_date = latest_assessment.date.strftime('%B %d, %Y')
        first_assessment_date = assessments[-1].date.strftime('%B %d, %Y')
        
        # Calculate risk trend
        if len(assessments) > 1:
            risk_change = latest_assessment.probability - assessments[1].probability
            risk_trend = "Improving" if risk_change < -0.05 else "Worsening" if risk_change > 0.05 else "Stable"
            risk_change = f"{abs(risk_change):.1%}"
        else:
            risk_trend = "Not enough data"
            risk_change = "N/A"
    else:
        current_risk_level = "No data"
        current_risk_class = ""
        latest_assessment_date = "No assessments yet"
        first_assessment_date = "N/A"
        risk_trend = "No data"
        risk_change = "N/A"
    
    # Prepare data for charts
    dates = [a.date.strftime('%Y-%m-%d') for a in reversed(assessments)]
    probabilities = [a.probability for a in reversed(assessments)]
    
    # Count risk factors
    risk_factor_counts = {}
    for assessment in assessments:
        factors = assessment.get_risk_factors()
        for factor in factors:
            risk_factor_counts[factor] = risk_factor_counts.get(factor, 0) + 1
    
    risk_factor_labels = list(risk_factor_counts.keys())
    risk_factor_counts = list(risk_factor_counts.values())
    
    return render_template('dashboard.html',
                         current_risk_level=current_risk_level,
                         current_risk_class=current_risk_class,
                         latest_assessment_date=latest_assessment_date,
                         total_assessments=total_assessments,
                         first_assessment_date=first_assessment_date,
                         risk_trend=risk_trend,
                         risk_change=risk_change,
                         dates=dates,
                         probabilities=probabilities,
                         risk_factor_labels=risk_factor_labels,
                         risk_factor_counts=risk_factor_counts,
                         recent_assessments=assessments[:5])

@app.route('/assessment')
@login_required
def assessment():
    return render_template('assessment.html')

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if request.method == 'POST':
        try:
            print("Form submitted, processing data...")
            # Get data from form
            data = {
                'age': int(request.form['age']),
                'sex': int(request.form['sex']),
                'cp': int(request.form['cp']),
                'trestbps': int(request.form['trestbps']),
                'chol': int(request.form['chol']),
                'fbs': int(request.form['fbs']),
                'restecg': int(request.form['restecg']),
                'thalach': int(request.form['thalach']),
                'exang': int(request.form['exang']),
                'oldpeak': float(request.form['oldpeak']),
                'slope': int(request.form['slope']),
                'ca': int(request.form['ca']),
                'thal': int(request.form['thal'])
            }
            print(f"Form data: {data}")
            
            # Create DataFrame
            input_data = pd.DataFrame([data])
            print("Input DataFrame:")
            print(input_data)
            
            # Load model
            model_path = Path("models/grid_search_best_model.joblib")
            if not model_path.exists():
                print("Model file not found")
                flash("Model file not found. Please ensure the model is properly installed.", "error")
                return redirect(url_for('assessment'))
            
            print("Loading model...")
            model = joblib.load(model_path)
            if model is None:
                print("Error loading model")
                flash("Error loading the model. Please try again later.", "error")
                return redirect(url_for('assessment'))
            print(f"Model loaded successfully. Model type: {type(model)}")
            
            # Preprocess data
            print("Preprocessing data...")
            try:
                preprocessed_data = preprocess_data(input_data)
                print("Preprocessed data shape:", preprocessed_data.shape)
                print("Preprocessed data:")
                print(preprocessed_data)
            except Exception as e:
                print(f"Error during preprocessing: {str(e)}")
                raise
            
            # Make prediction
            print("Making prediction...")
            try:
                prediction = model.predict(preprocessed_data)[0]
                probabilities = model.predict_proba(preprocessed_data)[0]
                probability = probabilities[1]
                print(f"Raw prediction: {prediction}")
                print(f"Probability array: {probabilities}")
                print(f"Heart disease probability: {probability}")
            except Exception as e:
                print(f"Error during prediction: {str(e)}")
                raise
            
            # Get risk assessment
            risk_level, risk_class = get_risk_level(probability)
            print(f"Risk level: {risk_level}, Risk class: {risk_class}")
            
            # Format risk factors with severity
            risk_factors_list = []
            raw_risk_factors = get_risk_factors(input_data.iloc[0])
            print(f"Identified risk factors: {raw_risk_factors}")
            for factor in raw_risk_factors:
                severity = "high" if any(high_risk in factor.lower() for high_risk in ["high", "significant", "advanced"]) else "normal"
                risk_factors_list.append({
                    "name": factor,
                    "description": get_factor_description(factor),
                    "severity": severity
                })
            
            # Format recommendations with priority
            recommendations_list = []
            raw_recommendations = get_recommendations(raw_risk_factors, risk_level)
            for rec in raw_recommendations:
                priority = "urgent" if any(urgent in rec.lower() for urgent in ["immediate", "urgent", "critical"]) else "normal"
                recommendations_list.append({
                    "title": rec,
                    "description": get_recommendation_description(rec),
                    "priority": priority
                })
            
            # Save assessment to database
            print("Saving assessment to database...")
            assessment = Assessment(
                user_id=current_user.id,
                risk_level=risk_level,
                probability=probability,
                risk_factors=json.dumps(raw_risk_factors),
                recommendations=json.dumps(raw_recommendations),
                input_data=json.dumps(data)
            )
            db.session.add(assessment)
            db.session.commit()
            print("Assessment saved to database")
            
            print("Rendering results template...")
            return render_template('results.html',
                                risk_level=risk_level,
                                risk_class=risk_class.lower(),
                                probability=probability,
                                risk_factors=risk_factors_list,
                                recommendations=recommendations_list,
                                assessment=assessment)
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            import traceback
            traceback.print_exc()  # Print the full traceback
            flash(f"Error during prediction: {str(e)}", "error")
            return redirect(url_for('assessment'))

def get_factor_description(factor):
    """Get detailed description for a risk factor"""
    descriptions = {
        "Advanced age": "Being over 60 years old increases the risk of heart disease.",
        "High blood pressure": "Blood pressure above 140 mmHg can damage your heart and blood vessels.",
        "High cholesterol": "Cholesterol levels above 240 mg/dl can lead to arterial blockages.",
        "Low maximum heart rate": "A low maximum heart rate during exercise may indicate reduced cardiac function.",
        "High maximum heart rate": "An unusually high heart rate during exercise may indicate cardiovascular stress.",
        "High fasting blood sugar": "Elevated blood sugar levels can damage blood vessels and nerves.",
        "Exercise-induced angina": "Chest pain during exercise indicates reduced blood flow to the heart.",
        "Significant ST depression": "ECG changes suggesting reduced blood flow to the heart muscle."
    }
    return descriptions.get(factor, "This factor contributes to your overall heart disease risk.")

def get_recommendation_description(recommendation):
    """Get detailed description for a recommendation"""
    descriptions = {
        "Regular blood pressure monitoring": "Check your blood pressure at least twice a week and keep a log.",
        "Consider reducing sodium intake": "Limit daily sodium intake to less than 2,300mg (about 1 teaspoon of salt).",
        "Regular cholesterol screening": "Get your cholesterol levels checked at least once every 6 months.",
        "Consider diet modifications": "Focus on a heart-healthy diet rich in fruits, vegetables, and whole grains.",
        "Regular blood sugar monitoring": "Monitor your blood sugar levels regularly and maintain a healthy diet.",
        "Consider consulting with a diabetes specialist": "Schedule an appointment with an endocrinologist for diabetes management.",
        "Immediate consultation with a cardiologist": "Schedule an urgent appointment with a cardiologist within the next few days.",
        "Consider stress test and detailed heart examination": "Request a comprehensive cardiac evaluation including stress test.",
        "Maintain current healthy lifestyle": "Continue your current health practices and regular check-ups.",
        "Regular check-ups as recommended by your doctor": "Follow your doctor's advice for routine health monitoring."
    }
    return descriptions.get(recommendation, "Follow this recommendation as part of your heart health management plan.")

@app.route('/download_report/<int:assessment_id>')
@login_required
def download_report(assessment_id):
    assessment = Assessment.query.get_or_404(assessment_id)
    if assessment.user_id != current_user.id:
        flash('Access denied', 'error')
        return redirect(url_for('dashboard'))
    
    # Create PDF
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        textColor=colors.HexColor('#dc3545'),
        alignment=1  # Center alignment
    )
    story.append(Paragraph("Heart Disease Risk Assessment Report", title_style))
    
    # Patient Info
    story.append(Paragraph("Patient Information", styles['Heading2']))
    story.append(Paragraph(f"Name: {current_user.name}", styles['Normal']))
    story.append(Paragraph(f"Assessment Date: {assessment.date.strftime('%B %d, %Y at %H:%M')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Risk Assessment
    risk_style = ParagraphStyle(
        'RiskLevel',
        parent=styles['Heading2'],
        fontSize=18,
        spaceAfter=20,
        textColor=colors.HexColor('#dc3545')
    )
    story.append(Paragraph("Risk Assessment", risk_style))
    story.append(Paragraph(f"Risk Level: {assessment.risk_level}", styles['Normal']))
    story.append(Paragraph(f"Probability: {assessment.probability:.1%}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Input Data
    story.append(Paragraph("Assessment Details", styles['Heading2']))
    input_data = json.loads(assessment.input_data)
    data_items = [
        ('Age', input_data['age']),
        ('Sex', 'Male' if input_data['sex'] == 1 else 'Female'),
        ('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'][input_data['cp']]),
        ('Resting Blood Pressure', f"{input_data['trestbps']} mm Hg"),
        ('Cholesterol', f"{input_data['chol']} mg/dl"),
        ('Fasting Blood Sugar', '> 120 mg/dl' if input_data['fbs'] == 1 else '≤ 120 mg/dl'),
        ('Resting ECG', ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'][input_data['restecg']]),
        ('Maximum Heart Rate', input_data['thalach']),
        ('Exercise Induced Angina', 'Yes' if input_data['exang'] == 1 else 'No'),
        ('ST Depression', f"{input_data['oldpeak']:.1f}"),
        ('ST Slope', ['Upsloping', 'Flat', 'Downsloping'][input_data['slope']]),
        ('Number of Vessels', input_data['ca']),
        ('Thalassemia', 'Normal' if input_data['thal'] == 3 else 'Fixed Defect' if input_data['thal'] == 6 else 'Reversible Defect' if input_data['thal'] == 7 else 'Unknown')
    ]
    
    # Create table for input data
    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f8f9fa')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#495057')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BOX', (0, 0), (-1, -1), 2, colors.black),
        ('LINEBELOW', (0, 0), (-1, 0), 2, colors.black),
    ])
    
    table_data = [['Parameter', 'Value']]
    table_data.extend(data_items)
    table = Table(table_data, colWidths=[doc.width/2.0]*2)
    table.setStyle(table_style)
    story.append(table)
    story.append(Spacer(1, 20))
    
    # Risk Factors
    story.append(Paragraph("Identified Risk Factors", styles['Heading2']))
    risk_factors = json.loads(assessment.risk_factors)
    for factor in risk_factors:
        bullet_style = ParagraphStyle(
            'Bullet',
            parent=styles['Normal'],
            leftIndent=20,
            firstLineIndent=-20,
            spaceBefore=5,
            spaceAfter=5
        )
        story.append(Paragraph(f"• {factor} - {get_factor_description(factor)}", bullet_style))
    story.append(Spacer(1, 20))
    
    # Recommendations
    story.append(Paragraph("Recommendations", styles['Heading2']))
    recommendations = json.loads(assessment.recommendations)
    for rec in recommendations:
        bullet_style = ParagraphStyle(
            'Bullet',
            parent=styles['Normal'],
            leftIndent=20,
            firstLineIndent=-20,
            spaceBefore=5,
            spaceAfter=5
        )
        story.append(Paragraph(f"• {rec} - {get_recommendation_description(rec)}", bullet_style))
    
    # Footer
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.gray,
        alignment=1  # Center alignment
    )
    story.append(Spacer(1, 30))
    story.append(Paragraph("This report is generated automatically and should be reviewed by a healthcare professional.", footer_style))
    story.append(Paragraph(f"Generated on {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}", footer_style))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    
    return send_file(
        buffer,
        as_attachment=True,
        download_name=f'heart_disease_assessment_{assessment.date.strftime("%Y%m%d_%H%M%S")}.pdf',
        mimetype='application/pdf'
    )

@app.route('/profile')
@login_required
def profile():
    assessments = Assessment.query.filter_by(user_id=current_user.id).all()
    return render_template('profile.html', assessments=assessments)

@app.route('/update_profile', methods=['POST'])
@login_required
def update_profile():
    if request.method == 'POST':
        current_user.email = request.form.get('email')
        db.session.commit()
        flash('Profile updated successfully!')
        return redirect(url_for('profile'))

@app.route('/assessment/<int:assessment_id>')
@login_required
def view_assessment(assessment_id):
    assessment = Assessment.query.get_or_404(assessment_id)
    
    # Check if the assessment belongs to the current user
    if assessment.user_id != current_user.id:
        flash('Access denied', 'error')
        return redirect(url_for('dashboard'))
    
    # Get risk factors and recommendations
    risk_factors_list = []
    raw_risk_factors = json.loads(assessment.risk_factors)
    for factor in raw_risk_factors:
        severity = "high" if any(high_risk in factor.lower() for high_risk in ["high", "significant", "advanced"]) else "normal"
        risk_factors_list.append({
            "name": factor,
            "description": get_factor_description(factor),
            "severity": severity
        })
    
    recommendations_list = []
    raw_recommendations = json.loads(assessment.recommendations)
    for rec in raw_recommendations:
        priority = "urgent" if any(urgent in rec.lower() for urgent in ["immediate", "urgent", "critical"]) else "normal"
        recommendations_list.append({
            "title": rec,
            "description": get_recommendation_description(rec),
            "priority": priority
        })
    
    return render_template('results.html',
                         risk_level=assessment.risk_level,
                         risk_class=assessment.risk_class.lower(),
                         probability=assessment.probability,
                         risk_factors=risk_factors_list,
                         recommendations=recommendations_list,
                         assessment=assessment)

if __name__ == '__main__':
    app.run(debug=True) 