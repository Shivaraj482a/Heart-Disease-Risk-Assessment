# Trained Models

This directory contains all the trained machine learning models used in the Heart Disease Risk Assessment System.

## Model Files

### Current Production Models
- `grid_search_best_model.joblib`: The main production model optimized using grid search
  - Location: `/content/drive/MyDrive/grid_search_best_model.joblib`
  - Currently used in production
  - Best performing model with optimized hyperparameters

### Decision Tree Models
- `decision_tree_3_gini.joblib`: Decision Tree with max_depth=3, criterion=gini
- `decision_tree_4_gini.joblib`: Decision Tree with max_depth=4, criterion=gini
- `decision_tree_5_entropy.joblib`: Decision Tree with max_depth=5, criterion=entropy
- `decision_tree_7_gini.joblib`: Decision Tree with max_depth=7, criterion=gini
- `decision_tree_None_entropy.joblib`: Decision Tree with unlimited depth, criterion=entropy

### Random Forest Models
- `random_forest_100_5.joblib`: RF with n_estimators=100, max_depth=5
- `random_forest_150_6.joblib`: RF with n_estimators=150, max_depth=6
- `random_forest_200_7.joblib`: RF with n_estimators=200, max_depth=7
- `random_forest_200_None.joblib`: RF with n_estimators=200, unlimited depth
- `random_forest_300_10.joblib`: RF with n_estimators=300, max_depth=10

### Gradient Boosting Models
- `gradient_boosting_100_0.1.joblib`: GB with n_estimators=100, learning_rate=0.1
- `gradient_boosting_150_0.1.joblib`: GB with n_estimators=150, learning_rate=0.1
- `gradient_boosting_200_0.05.joblib`: GB with n_estimators=200, learning_rate=0.05
- `gradient_boosting_250_0.05.joblib`: GB with n_estimators=250, learning_rate=0.05
- `gradient_boosting_300_0.01.joblib`: GB with n_estimators=300, learning_rate=0.01

### AdaBoost Models
- `adaboost_50_1.0.joblib`: AdaBoost with n_estimators=50, learning_rate=1.0
- `adaboost_100_0.5.joblib`: AdaBoost with n_estimators=100, learning_rate=0.5
- `adaboost_150_0.8.joblib`: AdaBoost with n_estimators=150, learning_rate=0.8
- `adaboost_200_0.1.joblib`: AdaBoost with n_estimators=200, learning_rate=0.1
- `adaboost_300_0.05.joblib`: AdaBoost with n_estimators=300, learning_rate=0.05

## Best Model Performance (grid_search_best_model.joblib)

The best model was selected through grid search with the following parameters:
```python
{
    'class_weight': 'balanced',
    'criterion': 'entropy',
    'max_depth': 7,
    'max_features': None,
    'min_samples_leaf': 4,
    'min_samples_split': 2,
    'splitter': 'random'
}
```

Performance Metrics:
- Accuracy: 0.833 (83.3%)
- ROC AUC: 0.858 (85.8%)
- Precision: 0.845 (84.5%)
- Recall: 0.741 (74.1%)

## Model Storage

All models are stored in Google Drive at `/content/drive/MyDrive/` for:
- Easy access
- Version control
- Backup purposes

## Model Selection Process

1. Initial testing of various algorithms:
   - Decision Trees with different depths and criteria
   - Random Forests with varying number of trees and depths
   - Gradient Boosting with different learning rates
   - AdaBoost with varying parameters

2. Grid Search Optimization:
   - Tested 7560 different parameter combinations
   - 5-fold cross-validation
   - Total of 37,800 fits
   - Selected best model based on balanced performance metrics

## Model Usage

The best model is loaded in `app.py` using:
```python
model_path = Path("models/grid_search_best_model.joblib")
model = joblib.load(model_path)
```

## Retraining

To retrain the models:
1. Use the UCI Heart Disease dataset
2. Run the training script: `python model_evaluation.py`
3. The new models will be saved in this directory

## Version Control

- Current Version: 1.0.0
- Last Updated: April 2024
- Changes: Initial release with comprehensive model comparison 