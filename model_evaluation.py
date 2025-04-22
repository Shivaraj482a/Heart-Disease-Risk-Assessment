import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report, 
    confusion_matrix
)
import joblib
import os
from google.colab import drive

def mount_google_drive():
    """Mount Google Drive and return True if successful"""
    try:
        drive.mount('/content/drive')
        print("Google Drive mounted successfully!")
        return True
    except Exception as e:
        print(f"Error mounting Google Drive: {str(e)}")
        return False

def load_model(model_path):
    """Load a trained model from Google Drive"""
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

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
    
    return X_scaled, y

def evaluate_model(model, X, y, model_name="Model"):
    """
    Evaluate model performance and generate visualizations
    """
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
        'roc_auc': roc_auc_score(y, y_pred)
    }
    
    # Print results
    print(f"\n{model_name} Performance Metrics:")
    print("=" * 50)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"ROC AUC Score: {metrics['roc_auc']:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print("=" * 50)
    print(classification_report(y, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
    plt.close()
    
    return metrics

def main():
    # Mount Google Drive
    if not mount_google_drive():
        return
    
    # Dictionary of models to test
    models_to_test = {
        'Grid Search Best Model': '/content/drive/MyDrive/grid_search_best_model.joblib',
        'Decision Tree (Depth 7)': '/content/drive/MyDrive/decision_tree_7_gini.joblib',
        'Random Forest (300 trees)': '/content/drive/MyDrive/random_forest_300_10.joblib',
        'Gradient Boosting': '/content/drive/MyDrive/gradient_boosting_100_0.1.joblib',
        'AdaBoost': '/content/drive/MyDrive/adaboost_100_0.5.joblib'
    }
    
    # Load and preprocess data
    try:
        print("\nLoading and preprocessing data...")
        X, y = load_and_preprocess_data()
        print("Data preprocessing completed!")
    except Exception as e:
        print(f"Error preprocessing data: {str(e)}")
        return
    
    # Test each model
    results = {}
    for model_name, model_path in models_to_test.items():
        print(f"\nTesting {model_name}...")
        model = load_model(model_path)
        if model is not None:
            try:
                metrics = evaluate_model(model, X, y, model_name)
                results[model_name] = metrics
            except Exception as e:
                print(f"Error evaluating {model_name}: {str(e)}")
    
    # Compare models
    if results:
        print("\nModel Comparison:")
        print("=" * 50)
        comparison_df = pd.DataFrame(results).round(4)
        print(comparison_df)
        
        # Plot comparison
        plt.figure(figsize=(12, 6))
        comparison_df.plot(kind='bar')
        plt.title('Model Performance Comparison')
        plt.xlabel('Metric')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('model_comparison.png')
        plt.close()

if __name__ == "__main__":
    main() 