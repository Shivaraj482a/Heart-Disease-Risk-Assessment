# Heart Disease Risk Assessment System

A comprehensive web application for assessing heart disease risk using machine learning. The system provides risk assessment, detailed analysis, and personalized recommendations based on patient data.

## Features

- User authentication and profile management
- Heart disease risk assessment using machine learning
- Detailed risk factor analysis
- Personalized recommendations
- PDF report generation
- Interactive dashboard with risk trends
- Responsive web interface

## Technology Stack

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **Database**: SQLite with SQLAlchemy
- **Machine Learning**: Scikit-learn
- **Authentication**: Flask-Login
- **PDF Generation**: ReportLab

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/heart-disease-risk-assessment.git
cd heart-disease-risk-assessment
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Initialize the database:
```bash
python create_db.py
```

5. Run the application:
```bash
python run.py
```

## Project Structure

```
heart-disease-risk-assessment/
├── app.py                 # Main Flask application
├── models.py             # Database models
├── forms.py              # Form definitions
├── requirements.txt      # Project dependencies
├── models/              # Trained ML models
│   ├── grid_search_best_model.joblib
│   └── scaler.joblib
├── static/              # Static files (CSS, JS)
├── templates/           # HTML templates
└── instance/           # Instance-specific files
```

## Model Information

The system uses a machine learning model trained on the UCI Heart Disease dataset. The model is optimized using grid search and includes:

- Feature engineering
- Robust scaling
- Cross-validation
- Hyperparameter optimization

## Usage

1. Register a new account or login
2. Navigate to the assessment page
3. Enter patient information
4. View the risk assessment results
5. Download detailed PDF report
6. Track risk trends in the dashboard

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- UCI Machine Learning Repository for the Heart Disease dataset
- Scikit-learn team for the machine learning tools
- Flask team for the web framework 