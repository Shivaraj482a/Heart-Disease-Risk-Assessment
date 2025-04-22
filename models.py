from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    assessments = db.relationship('Assessment', backref='user', lazy=True)
    date_registered = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Assessment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date = db.Column(db.DateTime, default=datetime.utcnow)
    risk_level = db.Column(db.String(20))
    probability = db.Column(db.Float)
    risk_factors = db.Column(db.Text)
    recommendations = db.Column(db.Text)
    input_data = db.Column(db.Text)

    @property
    def risk_class(self):
        """Return the appropriate CSS class based on risk level"""
        risk_classes = {
            'Very Low': 'success',
            'Low': 'info',
            'Moderate': 'warning',
            'High': 'warning',
            'Very High': 'danger'
        }
        return risk_classes.get(self.risk_level, 'secondary')

    def get_risk_factors(self):
        import json
        try:
            return json.loads(self.risk_factors)
        except:
            return []

    def get_recommendations(self):
        import json
        try:
            return json.loads(self.recommendations)
        except:
            return []

    def __repr__(self):
        return f'<Assessment {self.date}>' 